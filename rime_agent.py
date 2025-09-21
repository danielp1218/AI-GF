import asyncio
import logging
import random
import os
from dotenv import load_dotenv

from livekit.agents import (
    AgentSession,
    AutoSubscribe,
    JobContext,
    JobProcess,
    tts,
    metrics,
    RoomInputOptions,
    RoomOutputOptions,
    WorkerOptions,
    cli,
    ChatContext,
    ChatMessage,
    function_tool,
    RunContext
)
from livekit.agents.voice import Agent, MetricsCollectedEvent
from livekit.plugins import (
    openai,
    noise_cancellation,
    rime,
    silero,
    tavus
)
from livekit.agents.tokenize import tokenizer

from livekit.plugins.turn_detector.multilingual import MultilingualModel

from agent_configs import VOICE_CONFIGS
from memory_utils import Memory

load_dotenv()
logger = logging.getLogger("voice-agent")

VOICE_NAMES = ["celeste"]
# randomly select a voice from the list
VOICE = random.choice(VOICE_NAMES)

def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()

class RimeAssistant(Agent):
    def __init__(self, memory: Memory) -> None:
        super().__init__(instructions=VOICE_CONFIGS[VOICE]["llm_prompt"])
        self.memory = memory
        logger.info("RimeAssistant initialized")
    
    @function_tool()
    async def recall_memories(
        self,
        context: RunContext,
        query: str,
        top_k: int = 3
    ) -> str:
        logger.info(f"=== recall_memories TOOL CALLED ===")
        logger.info(f"Query: {query}, top_k: {top_k}")
        
        try:
            # Limit top_k to reasonable bounds
            top_k = max(1, min(top_k, 10))
            
            # Retrieve relevant memories using RAG
            relevant_memories = self.memory.get_relevant_memories(query, top_k=top_k)
            
            logger.info(f"Found {len(relevant_memories)} memories")
            
            if relevant_memories:
                # Build memory context for RAG with high-confidence memories only
                high_confidence_memories = []
                for memory in relevant_memories:
                    score = memory.get('score', 0)
                    if score > 0.6:  # Higher threshold for better quality
                        high_confidence_memories.append({
                            'text': memory['text'],
                            'score': score,
                            'timestamp': memory.get('timestamp', 'unknown')
                        })
                
                if high_confidence_memories:
                    memory_context = f"Found {len(high_confidence_memories)} relevant memories:\n\n"
                    for i, memory in enumerate(high_confidence_memories, 1):
                        memory_context += f"{i}. {memory['text']} (confidence: {memory['score']:.2f})\n"
                    
                    memory_context += f"\nUse these memories to personalize your response to the user."
                    
                    logger.info(f"Returning {len(high_confidence_memories)} high-confidence memories")
                    return memory_context
                else:
                    logger.debug("No high-confidence memories found")
                    return f"I found {len(relevant_memories)} memories but they weren't confident enough matches for your query '{query}'. You might want to try a more specific or different query."
            else:
                logger.info("No memories found for this query")
                return f"I couldn't find any memories related to '{query}'. This might be the first time we're discussing this topic."
                
        except Exception as e:
            logger.warning(f"Memory retrieval failed: {e}")
            import traceback
            traceback.print_exc()
            return f"I had trouble accessing my memories right now. Let me respond based on our current conversation."
async def entrypoint(ctx: JobContext):
    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)

    # Wait for the first participant to connect
    participant = await ctx.wait_for_participant()

    logger.info(f"Running Rime voice agent for voice config {VOICE} and participant {participant.identity}")

    rime_tts = rime.TTS(
        **VOICE_CONFIGS[VOICE]["tts_options"]
    )
    if VOICE_CONFIGS[VOICE].get("sentence_tokenizer"):
        sentence_tokenizer = VOICE_CONFIGS[VOICE].get("sentence_tokenizer")
        if not isinstance(sentence_tokenizer, tokenizer.SentenceTokenizer):
            raise TypeError(
                f"Expected sentence_tokenizer to be an instance of tokenizer.SentenceTokenizer, got {type(sentence_tokenizer)}"
            )
        rime_tts = tts.StreamAdapter(tts=rime_tts, sentence_tokenizer=sentence_tokenizer)


    memory = Memory()

    agent = RimeAssistant(memory)

    session = AgentSession(
        stt=openai.STT(),
        llm=openai.LLM(model="gpt-4o-mini"),
        tts=rime_tts,
        vad=ctx.proc.userdata["vad"],
        turn_detection=MultilingualModel()
    )
    usage_collector = metrics.UsageCollector()

    # Add memory integration through conversation events
    @session.on("conversation_item_added")
    def _on_conversation_item_added(ev):
        item = ev.item
        
        # Only process final (not interrupted) messages
        if not item.interrupted:
            if item.role == "user" and item.text_content:
                logger.info(f"Storing user message in memory: {item.text_content[:50]}...")
                try:
                    asyncio.create_task(
                        asyncio.to_thread(memory.analyze_and_memorize, item.text_content, "User message")
                    )
                except Exception as e:
                    logger.warning(f"Failed to store memory: {e}")
            elif item.role == "assistant" and item.text_content:
                logger.debug(f"Assistant response logged")

    @session.on("metrics_collected")
    def _on_metrics_collected(ev: MetricsCollectedEvent):
        metrics.log_metrics(ev.metrics)
        usage_collector.collect(ev.metrics)

    async def log_usage():
        summary = usage_collector.get_summary()
        logger.info(f"Usage: {summary}")

    ctx.add_shutdown_callback(log_usage)

    persona_id = os.getenv("TAVUS_PERSONA_ID")
    replica_id = os.getenv("TAVUS_REPLICA_ID")

    # --- Tavus integration ---
    avatar = tavus.AvatarSession(
        replica_id=replica_id,      # Replace with your actual replica ID
        persona_id=persona_id,      # Replace with your actual persona ID
        # Optional: avatar_participant_name="Tavus-avatar-agent"
    )
    await avatar.start(session, room=ctx.room)
    # -------------------------

    await session.start(
        room=ctx.room,
        agent=agent,  # Pass the agent instance
        room_input_options=RoomInputOptions(
            noise_cancellation=noise_cancellation.BVC()
        ),
        room_output_options=RoomOutputOptions(
            audio_enabled=False  # Tavus handles audio separately
        )
    )

    await session.say(VOICE_CONFIGS[VOICE]["intro_phrase"])

    ctx.add_shutdown_callback(session.aclose)

if __name__ == "__main__":
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
            prewarm_fnc=prewarm,
        ),
    )
