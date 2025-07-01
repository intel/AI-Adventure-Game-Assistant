import os
import sys
import multiprocessing
from async_worker import AsyncWorker

class VADWorker(AsyncWorker):
    def __init__(self):
        super().__init__()
        
        self.result_queue = multiprocessing.Queue()
 
    def _work_loop(self):
        import sherpa_onnx
        import sounddevice as sd
        
        mic_sample_rate = 16000
        if "SHERPA_ONNX_MIC_SAMPLE_RATE" in os.environ:
            mic_sample_rate = int(os.environ.get("SHERPA_ONNX_MIC_SAMPLE_RATE"))
            print(f"Change microphone sample rate to {mic_sample_rate}")

        sample_rate = 16000
        seconds_per_read = 0.2 # 0.2 second = 200 ms
        samples_per_read = int(seconds_per_read * sample_rate)  

        config = sherpa_onnx.VadModelConfig()
        config.silero_vad.model = "models/silero_vad.onnx"
        config.sample_rate = sample_rate

        vad = sherpa_onnx.VoiceActivityDetector(config, buffer_size_in_seconds=30)
        
        inactivity_cache = []
        inactivity_cachei = 0
        
        activity_cache = []
        
        # this basically defines how much 'inactivity' is kept as starting context
        # for the 'active' audio. VAD many times clips the first word or so of the 
        # input, and so passing a bit of the earlier bits that were considered 'inactive'
        # help us to regain some of that back.. 
        # Audio in seconds = max_inactivity_cache_len(5) * samples_per_read (0.2 secs) = 1 second of 'inactivity' prepended.
        max_inactivity_cache_len = 5
        
        devices = sd.query_devices()
        if len(devices) == 0:
            print("No microphone devices found")
            print(
                "If you are using Linux and you are sure there is a microphone "
                "on your system, please use "
                "./vad-alsa.py"
            )
            sys.exit(0)
        
        #uncomment this to print all detected input devices, and map.
        #print(devices)
        
        input_device_idx = sd.default.device[0]
        print(f'VAD: Using default device: {devices[input_device_idx]["name"]}')

        print("VAD is now listening...")
        
        #absolute time in seconds
        absolute_time = 0
        
        printed = False
        try:
            with sd.InputStream(
                channels=1, dtype="float32", samplerate=mic_sample_rate
            ) as s:
                while self._running.value:
                    samples, _ = s.read(samples_per_read)  # a blocking read
                    samples = samples.reshape(-1)
                    absolute_time += seconds_per_read
                   
                    if mic_sample_rate != sample_rate:
                        import librosa

                        samples = librosa.resample(
                            samples, orig_sr=mic_sample_rate, target_sr=sample_rate
                        )

                    vad.accept_waveform(samples)

                    if vad.is_speech_detected() and not printed:
                        print("Detected speech")
                        printed = True
                        
                    if printed:
                        activity_cache.append(samples)

                    if not vad.is_speech_detected():
                        
                        if not printed:
                            if len(inactivity_cache) < max_inactivity_cache_len:
                                inactivity_cache.append(samples)
                            else:
                                inactivity_cache[inactivity_cachei] = samples
                                inactivity_cachei = (inactivity_cachei + 1) % max_inactivity_cache_len
         
                        printed = False

                    while not vad.empty():
                        to_transcribe_list = []
                        
                        for i in range(0, len(inactivity_cache)):
                            to_transcribe_list.append(inactivity_cache[inactivity_cachei])
                            inactivity_cachei = (inactivity_cachei + 1) % max_inactivity_cache_len
                        
                        to_transcribe_list += activity_cache
 
                        segment_length_in_seconds = len(to_transcribe_list)*seconds_per_read
                        start_time = absolute_time - segment_length_in_seconds
                        self.result_queue.put((start_time, segment_length_in_seconds, to_transcribe_list,))
                        
                        #reset our caches
                        inactivity_cache = []
                        inactivity_cachei = 0
                        activity_cache = []
                        
                        vad.pop()
        except KeyboardInterrupt:
            print("\nVADWorker: Caught Ctrl + C. Exit")
                        

from queue import Empty                    
class WhisperWorker(AsyncWorker):
    def __init__(self, input_queue, device, ui_update_queue):
        super().__init__()
        
        self.input_queue = input_queue
        self.result_queue = multiprocessing.Queue()
        self.device = device
        self.ui_update_queue = ui_update_queue
        
    def _work_loop(self):
        import numpy as np
        import openvino_genai
        
        print("Creating whisper pipeline to run on device=", self.device)    
        worker_log = f"<b>ASR: Loading Whisper to <span style=\"color: green;\">{self.device}</span>...</b><br>"
        self.ui_update_queue.put(("worker_log", worker_log,))
        pipe = openvino_genai.WhisperPipeline("models/whisper-base", self.device)
        whisper_config = openvino_genai.WhisperGenerationConfig(
            "models/whisper-base" + "/generation_config.json"
        )
        whisper_config.max_new_tokens = 256  # increase this based on your speech length
        # 'task' and 'language' parameters are supported for multilingual models only
        whisper_config.language = "<|en|>"  # can switch to <|zh|> for Chinese language
        whisper_config.task = "transcribe"
        whisper_config.return_timestamps = False
        worker_log = f"<b>ASR: Loading Whisper to <span style=\"color: green;\">{self.device}</span>... [DONE]</b><br>"
        self.ui_update_queue.put(("worker_log", worker_log,))
        self.ui_update_queue.put(("ready", 1,))
        print("done creating whisper LLM")
        
        while self._running.value:
            try:
                queue_entry = self.input_queue.get(timeout=1)
                start_time = queue_entry[0]
                segment_length = queue_entry[1]
                to_transcribe_list = queue_entry[2]
                
                activity_array = np.concatenate(to_transcribe_list)
                result = pipe.generate(activity_array, whisper_config)
                transcription=str(result)
                
                self.result_queue.put((start_time, segment_length, transcription))
                
                print(transcription)
                
            except Empty:
                continue  # Queue is empty, just wait
            

def test_main():
    vad_worker = VADWorker()
    vad_result_queue = vad_worker.result_queue
    vad_worker.start()
    
    whisper_worker = WhisperWorker(vad_result_queue)
    whisper_worker.start()

    try:
        # Just a loop to allow the workers to do their thing, and wait for user
        # to Ctrl-C
        while True:
            import time
            time.sleep(1)
            #print("got result!")
    except KeyboardInterrupt:
        print("Main: Stopping workers...")
        vad_worker.stop() 
        whisper_worker.stop()        


if __name__ == "__main__":
    test_main()