import openvino_genai as ov_genai
import json

LLM_ORCHESTRATOR_SYSTEM_MESSAGE_SHORT="""
You are an AI assistant tasked with identifying scenes, locations, or moments in a tabletop adventure game that are relevant and worth illustrating.
"""

def extract_json_content(s):
    start = s.find('{')
    end = s.rfind('}')
    if start == -1 or end == -1 or start > end:
        return None  # No valid JSON-like content found
    return s[start:end+1]


class LLMOrchestrator:
    def __init__(self, llm_pipeline, ui_update_queue, theme):
        self.llm_pipeline =  llm_pipeline
        self.ui_update_queue = ui_update_queue

        generate_config = ov_genai.GenerationConfig()
        generate_config.temperature = 0.0
        #enerate_config.top_p = 0.95
        generate_config.max_length = 2048
        generate_config.apply_chat_template = False

        self.generate_config = generate_config

        self.meaningful_message_pairs = []
        self.system_message = self.create_system_message(theme)

        self.current_location = 'None'
        self.current_rulebook_query = 'None'
        self.theme = theme

        self.current_illustration_focus = None

    def create_system_message(self, theme):
        return LLM_ORCHESTRATOR_SYSTEM_MESSAGE_SHORT

    def llm_streamer(self, subword):
        print(subword, end='', flush=True)
        self.stream_message += subword

        search_string = "SD Prompt:"
        if search_string in self.stream_message and 'None' not in self.stream_message:
            if self.stream_sd_prompt_index is None:
                self.stream_sd_prompt_index = self.stream_message.find(search_string)

            start_index = self.stream_sd_prompt_index
            # Calculate the start index of the new string (1 character past the ':')
            prompt = self.stream_message[start_index + len(search_string):].strip()

            self.ui_update_queue.put(("caption", prompt,))
            #self.caption_updated.emit(prompt)

        elif 'None' in self.stream_message:
            #Sometimes the LLM gives a response like: None (And then some long description why in parenthesis)
            # Basically, as soon as we see 'None', just stop generating tokens.
            return True

        # Return flag corresponds whether generation should be stopped.
        # False means continue generation.
        return False

    def llm_streamer_new(self, subword):
         print(subword, end='', flush=True)
         return False

    def run(self, raw_transcript_previous, raw_transcript_new):
        self.ui_reasoning_state = ""
        self.update_ui_reasoning_state("<b>[Illustrator]</b> <br>")
        self.llm_pipeline.finish_chat()
        self.llm_pipeline.start_chat(system_message=self.system_message)
        

        if len(raw_transcript_previous)  > 0:
            user_message = "For context, here is some bits of a transcript from our live tabletop adventure game that we've already discussed:\n"
            user_message += "[ Previous Context Transcript ]:"+ "\n"
            for t in raw_transcript_previous:
                user_message += t

            user_message += "I have only provided that 'Previous Context Transcript' to give some context to the bits of 'New Transcript' that I am now presenting to you now:"
            user_message += "\n[ New Transcript ]:"+ "\n"
            for t in raw_transcript_new:
                user_message += t

            user_message += "I will be asking you a series of questions about the content found in 'New Transcript'.\n"
            user_message += "You can use the 'Previous Context Transcript' as context to the content given in 'New Transcript'.\n"
        else:
            user_message = "Here are some bits of a transcript from our live tabletop adventure game currently being played:"
            user_message += "\n[ New Transcript ]:"+ "\n"
            for t in raw_transcript_new:
                user_message += t
            user_message += "I will be asking you a series of questions about the content found in 'New Transcript'.\n"


        def yes_reponse(llm_response):
            llm_response = llm_response.lower()
            search_string = "answer:"

            if search_string in llm_response:
                # Find the start of the search string
                start_index = llm_response.find(search_string)
                # Calculate the start index of the new string (1 character past the ':')
                answer = llm_response[start_index + len(search_string):].strip()
                if "yes" in answer:
                    return True

            return False

        if False:
            user_message += "Question 1: Does it seem like *any* content / conversation found within 'New Transcript' is relevent to the tabletop session being played?\n"
            user_message += "I'd like you to think about it and then give me a 'yes' or 'no' answer.\n"
            user_message += "If you think that any part of the content given in 'New Transcript' is relevant, say 'yes'. Otherwise, if it seems totally unrelevant, say 'no.'\n"
            user_message += "For example, I want you to output something in the following format. A 'Reason' followed by an 'Answer':\n"
            user_message += "Reason: <Short (25 words or less) reason for my yes or no answer>\n"
            user_message += "Answer: <'yes' or 'no'>"


            print("***********************************************************************************")
            print(user_message)
            print("***********************************************************************************")

            llm_result = self.llm_pipeline.generate(inputs=user_message, generation_config=self.generate_config, streamer=self.llm_streamer_new)
            print("")

            if not yes_reponse(llm_result):
                return None

            user_message = ""

        user_message +="""
        Given the 'New Transcript' segment above, determine if it contains the description of a scene that is relevant to the in-game story.
        Respond with "Yes" if it meets these criteria, or "No" if it does not, along with a reason.
        For example, I want you to output something in the following format. A 'Reason' followed by an 'Answer':
        Reason: <Short (25 words or less) reason for my yes or no answer>
        Answer: <'yes' or 'no'>
        """

        print("***********************************************************************************")
        print(user_message)
        print("***********************************************************************************")
        self.update_ui_reasoning_state(" [Is there a description of a relevant scene?]   ....")
        
        llm_result = self.llm_pipeline.generate(inputs=user_message, generation_config=self.generate_config, streamer=self.llm_streamer_new)
        print("")

        if not yes_reponse(llm_result):
            self.update_ui_reasoning_state("   <span style=\"color: red;\">[NO]</span> <br>")
            return None
            
        self.update_ui_reasoning_state("   <span style=\"color: green;\">[YES]</span> <br>")

        user_message = """
        You have determined that there is a scene worth illustrating. Please provide a brief description of the key elements or focus of what should be illustrated.
        If now that you think about it, 'New Transcript' doesn't contain the description of something is relevant to the in-game story, just return the word 'None'.
        I'd like you to give your response in this format"
        Illustration Focus: <Some brief description of the focus of what should be illustrated, 10 words or less, or 'None'>"
        """
        
        print("***********************************************************************************")
        print(user_message)
        print("***********************************************************************************")
        self.update_ui_reasoning_state(" [ILLUSTRATION FOCUS]:<br>  ")
        llm_result = self.llm_pipeline.generate(inputs=user_message, generation_config=self.generate_config, streamer=self.llm_streamer_new)
        
        #print(llm_result)
        search_string = "Illustration Focus:"

        #sometimes the llm will return 'SD Prompt: None', so filter out that case.
        if search_string in llm_result and 'None' not in llm_result:
            # Find the start of the search string
            start_index = llm_result.find(search_string)
            # Calculate the start index of the new string (1 character past the ':')
            new_focus = llm_result[start_index + len(search_string):].strip()
            #print(f"Extracted prompt: '{prompt}'")
            self.update_ui_reasoning_state(f" {new_focus}<br>")
        else:
            self.update_ui_reasoning_state(" NONE <br>")
            return None

        if self.current_illustration_focus is not None:
            
            user_message = f"""
            You have identified a new focus for illustration based on the 'New Transcript' segment.
            Compare this new focus with the previous illustration focus to determine if it worth replacing the previous illustration with a new one.
            Basically, if the main focus for illustration has changed, and it makes sense for us to illustrate, reply with 'yes'.
            Previous Illustration Focus: \"{self.current_illustration_focus}\"
            New Illustration Focus: \"{new_focus}\"
            For example, I want you to output something in the following format. A 'Reason' followed by an 'Answer':
            Reason: <Short (25 words or less) reason for my yes or no answer>
            Answer: <'yes' or 'no', whether you think the new focus is worth illustrating>
            """

            print("***********************************************************************************")
            print(user_message)
            print("***********************************************************************************")
            self.update_ui_reasoning_state(" [REPLACE CURRENT ILLUSTRATION? ]                ....")
            llm_result = self.llm_pipeline.generate(inputs=user_message, generation_config=self.generate_config, streamer=self.llm_streamer_new)
            print("")

            if not yes_reponse(llm_result):
                self.update_ui_reasoning_state("   <span style=\"color: red;\">[NO]</span> <br>")
                return None
            else:
                self.update_ui_reasoning_state("   <span style=\"color: green;\">[YES]</span> <br>")

        
        self.current_illustration_focus = new_focus

        user_message = f"""
        You have determined that the new illustration focus is distinct and worth illustrating.
        Please generate a detailed prompt for Stable Diffusion that will be used to generate the image.
        Don't include any details about the players or characters as I want to only illustrate their surrounding, or the specific focus.
        Include any important details that would help an artist understand what to depict.
        I'd like you to give your response in this format:
        SD Prompt: <Some detailed prompt that can be used to pass to stable diffusion to ilustrate the scene. 30 words or less>
        """

        print("***********************************************************************************")
        print(user_message)
        print("***********************************************************************************")

        self.stream_message = ""
        self.stream_sd_prompt_index = None
        self.update_ui_reasoning_state(" [ GENERATING ILLUSTRATION DETAILS ]             ....")
        llm_result = self.llm_pipeline.generate(inputs=user_message, generation_config=self.generate_config, streamer=self.llm_streamer)
        self.update_ui_reasoning_state("   <span style=\"color: green;\">[DONE]</span> <br>")
        print("")
        #print(llm_result)
        search_string = "SD Prompt:"

        #sometimes the llm will return 'SD Prompt: None', so filter out that case.
        if search_string in llm_result and 'None' not in llm_result:
            # Find the start of the search string
            start_index = llm_result.find(search_string)
            # Calculate the start index of the new string (1 character past the ':')
            prompt = llm_result[start_index + len(search_string):].strip()
            #print(f"Extracted prompt: '{prompt}'")

            return prompt



        return None

    def update_ui_reasoning_state(self, blurb):
        self.ui_reasoning_state += blurb
        self.ui_update_queue.put(("update_reasoning_state", self.ui_reasoning_state,))
    