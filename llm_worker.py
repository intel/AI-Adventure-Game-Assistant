from async_worker import AsyncWorker
from queue import Empty
from multiprocessing import Queue

import time
def seconds_to_hms(seconds):
    return "[" + time.strftime("%H:%M:%S", time.gmtime(seconds)) + "]"

# A async (subprocess) class that runs anything LLM-related (orchestrator, subworkers, etc.)
# They should all be invoked from within a single instance of this class, as we ideally want
# orchestrator and subagents to all share the same instance of a GenAI LLM Pipeline.
# (as each instance of an LLM Pipeline would need dedicated set of weights loaded into memory)
class LLMWorker(AsyncWorker):
    def __init__(self, transcription_queue, sd_prompt_queue, ui_update_queue, llm_device, theme):
        super().__init__()

        self.transcription_queue = transcription_queue
        self.sd_prompt_queue = sd_prompt_queue
        self.llm_device = llm_device
        self.theme = theme
        self.ui_update_queue = ui_update_queue

    def _work_loop(self):
        import openvino_genai as ov_genai

        print("Creating an llm pipeline to run on ", self.llm_device)
        
        worker_log = f"<b>LLM: Loading llama-3.1-8b-instruct to <span style=\"color: green;\">{self.llm_device}</span>...</b><br>"
        self.ui_update_queue.put(("worker_log", worker_log,))
        #llm_model_path = r"./models/llama-3.1-8b-instruct/INT4_compressed_weights"
        #llm_model_path = r"./models/llama-3.2-3b-instruct/INT4_compressed_weights"
        llm_model_path = r"./models/llama-3.1-8b-instruct-awq/INT4_compressed_weights"
        #llm_model_path = r"./models/llama-3.2-3b-instruct-awq/INT4_compressed_weights"
        llm_device = self.llm_device

        if llm_device == 'NPU':
            pipeline_config = {"MAX_PROMPT_LEN": 1536, "NPUW_CACHE_DIR": ".npucache", "GENERATE_HINT": "BEST_PERF"}
            llm_pipe = ov_genai.LLMPipeline(llm_model_path, llm_device, pipeline_config)
        else:
            llm_pipe = ov_genai.LLMPipeline(llm_model_path, llm_device)
        
        
        print("llm pipeline has been created.")
        worker_log = f"<b>LLM: Loading llama-3.1-8b-instruct to <span style=\"color: green;\">{self.llm_device}</span>... [DONE]</b><br>"
        self.ui_update_queue.put(("worker_log", worker_log,))

        from llm_processing.orchestrator import LLMOrchestrator

        orchestrator = LLMOrchestrator(llm_pipe, self.ui_update_queue, self.theme)

        transcription_history = []
        TRANSCRIPT_CONTEXT_DURATION_SECS = 30
        self.ui_update_queue.put(("ready", 1,))
        while self._running.value:
            try:
                #self.progress_updated.emit(0, "listening")

                def queue_entry_processing(queue_entry):
                    start_time = queue_entry[0]
                    segment_length = queue_entry[1]
                    transcription = queue_entry[2]
                    transcription_history.append((start_time, segment_length, transcription, False,))

                # Each queue entry is a tuple:
                # (start_time_in_seconds, segment_length_in_seconds, transcription)
                queue_entry = self.transcription_queue.get(timeout=1)
                queue_entry_processing(queue_entry)

                # There might be more than one bit of transcript that have been added to the queue while LLM processing was taking place.
                # Add all of them to the current processing...
                while True:
                    try:
                        queue_entry = self.transcription_queue.get_nowait()
                        queue_entry_processing(queue_entry)
                    except Empty:
                        break

                # starting with the most recent entry in our transcription_history,
                # build up the current transcription.
                new_transcription_content_seconds = 0
                overall_context_length_in_seconds = 0
                current_transcription_history_i = len(transcription_history) - 1
                raw_transcript_previous = []
                raw_transcript_new = []
                while current_transcription_history_i >= 0:
                    transcription_history_entry = transcription_history[current_transcription_history_i]
                    start_time = transcription_history_entry[0]
                    segment_length = transcription_history_entry[1]
                    transcription = transcription_history_entry[2]
                    already_used = transcription_history_entry[3]

                    overall_context_length_in_seconds += segment_length

                    if not already_used:
                        raw_transcript_new.insert(0,f"    {seconds_to_hms(start_time)}: \"{transcription}\"" + "\n")
                        new_transcription_content_seconds += segment_length
                    else:
                        raw_transcript_previous.insert(0,f"    {seconds_to_hms(start_time)}: \"{transcription}\"" + "\n")
                        if overall_context_length_in_seconds >= TRANSCRIPT_CONTEXT_DURATION_SECS:
                            break

                    current_transcription_history_i = current_transcription_history_i - 1

                # we need at least 5 more seconds of 'new' content in order to invoke the LLM again..
                if new_transcription_content_seconds < 3:
                    continue

                # mark all as used
                current_transcription_history_i = len(transcription_history) - 1
                while current_transcription_history_i >= 0:
                    transcription_history_entry = transcription_history[current_transcription_history_i]

                    # as soon as we find one marked as True, all before it must already be
                    # marked as True as well, so just break outta here.
                    if transcription_history_entry[3]:
                        break

                    transcription_history[current_transcription_history_i] = (
                    transcription_history_entry[0], transcription_history_entry[1],
                    transcription_history_entry[2], True,
                    )
                    current_transcription_history_i = current_transcription_history_i - 1

                # assemmble user prompt
                #user_message = "Current Location: " + current_location + "\n"
                #user_message += "Current Rulebook Query: "  + current_rulebook_query + "\n"
                user_message = "[ Previous Context Transcript ]:"+ "\n"
                for t in raw_transcript_previous:
                    user_message += t

                if len(raw_transcript_previous) == 0:
                    user_message += "    (No Previous Context)"+ "\n"

                user_message += "\n[ New Transcript ]:"+ "\n"
                for t in raw_transcript_new:
                    user_message += t

                #transcription = queue_entry[2]
                #transcription = f"\"{transcription}\""
                self.ui_update_queue.put(("progress", "processing",))
                orchestrator_result = orchestrator.run(raw_transcript_previous=raw_transcript_previous, raw_transcript_new=raw_transcript_new)
                self.ui_update_queue.put(("progress", "listening",))
                #Right now, it's either a string (to run SD), or it's None.
                # This will change to a dictionary once the orchestrator turns into
                # an actual orchestrator.
                if orchestrator_result is not None:
                    self.sd_prompt_queue.put(orchestrator_result)

            except Empty:
                continue  # Queue is empty, just wait

def test_main():
    sd_prompt_queue = Queue()
    transcription_queue = Queue()
    ui_update_queue = Queue()

    llm_worker = LLMWorker(transcription_queue, sd_prompt_queue, ui_update_queue, "GPU", "Medieval Fantasy Adventure")
    llm_worker.start()

    try:
        while True:
            try:
                transcription = input('')
            except EOFError:
                break

            transcription_queue.put((0, 1, transcription))

    except KeyboardInterrupt:
        print("Main: Stopping workers...")
        llm_worker.stop()

if __name__ == "__main__":
    test_main()