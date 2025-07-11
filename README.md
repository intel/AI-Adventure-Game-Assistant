<div id="top" align="center">
  <h1>AI Adventure Game Assistant</h1>

</div>

[![Apache License Version 2.0](https://img.shields.io/badge/license-Apache_2.0-green.svg)](https://github.com/openvinotoolkit/openvino_build_deploy/blob/master/LICENSE.txt)

The project integrates image creation with generative AI, voice activity detection (VAD), automatic speech recognition (ASR), large language models (LLMs), and natural language processing (NLP). A live voice transcription pipeline is connected to an LLM, which makes intelligent decisions about whether the user is describing the scene to an adventure game. When the LLM detects a new scene, the LLM will produce a detailed text prompt suitable for stable diffusion, which the application uses to illustrate the image. Utilizing the OpenVINO™ GenAI framework, this kit demonstrates the use of text2image, LLM pipeline, and whisper speech2text APIs.

This project uses the following technology stack:
- [OpenVINO Toolkit](https://www.intel.com/content/www/us/en/developer/tools/openvino-toolkit/overview.html) ([docs](https://docs.openvino.ai/))
- [OpenVINO GenAI](https://github.com/openvinotoolkit/openvino.genai)
- [Whisper](https://github.com/openai/whisper)
- [Llama3-8b-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct)
- [Single Image Super Resolution](https://arxiv.org/abs/1807.06779)
- [Latent Consistency Models](https://arxiv.org/abs/2310.04378)



![ai_adventure_experience_desert](https://github.com/user-attachments/assets/2144ae33-9e41-4e48-9992-ddec17ef5579)

Contributors: Ryan Metcalfe, Garth Long, Arisha Kumar

### What's New

New updates will be added here.

<details open><summary><b>Table of Contents</b></summary>
  
- [Getting Started](#getting-started)
  - [Installing Prerequisites](#installing-prerequisites)
  - [Setting Up Your Environment](#setting-up-your-environment)
  - [Running the Application](#running-the-application)
- [Additional Resources](#additional-resources)

</details>

# Getting Started
Now, let's dive into the steps starting with installing Python. 

## Installing Prerequisites

Now, let's dive into the steps starting with installing Python. This project requires Python 3.10 or higher and a few libraries. If you don't have Python installed on your machine, go to https://www.python.org/downloads/ and download the latest version for your operating system. Follow the prompts to install Python, making sure to check the option to add Python to your PATH environment variable.

Install libraries and tools:

If you're using Ubuntu, install required dependencies like this:
```shell
sudo apt install git git-lfs gcc python3-venv python3-dev portaudio19-dev
```
_NOTE: If you are using Windows, you will probably need to install [Microsoft Visual C++ Redistributable](https://aka.ms/vs/16/release/vc_redist.x64.exe) also._

## Setting Up Your Environment
### Cloning the Repository and Installing Dependencies

To clone the repository, run the following command:

```shell
git clone https://github.com/intel/AI-Adventure-Game-Assistant.git
```

The above will clone the repository into a directory named "AI-Adventure-Game-Assistant" in the current directory. Then, navigate into the directory using the following command:

```shell
cd AI-Adventure-Game-Assistant
```

Next the below will create a virtual environment, activate the environment, and install the required dependencies for the setup and execution of the project.

Linux:
```shell
python3 -m venv run_env
source run_env/bin/activate
pip install -r requirements.txt
```

Windows:
```shell
python -m venv run_env
run_env\Scripts\activate
pip install -r requirements.txt
``` 

### Downloading and Preparing Models
Next, you’ll download and optimize the required models via the running of a download script. 

- Whisper: Speech recognition
- Llama3-8b-instruct: Intelligent LLM helper
- Latent Consistency Models: Image generation
- Super Resolution: Increase the resolution of the generated image

To use the Llama model, you need to have a HuggingFace account and generate a token for logging in on the command line.
1. Create an account on  [HuggingFace.co](https://huggingface.co)
2. Create an access token by going to [Settings > Access Tokens](https://huggingface.co/settings/tokens) and clicking +Create new token 
3. Select Token Type as `Read` and give a name
4. Copy the generated token, being sure to save it somewhere for later, if needed. 

To run the download script:
```shell
huggingface-cli login 
# paste the token when prompted.
python download_and_prepare_models.py
cd ..
```
## Running the Application

Open a terminal or you can use the existing one with `run_env` environment activated and start the GUI - <br>

```shell
python app.py 
```

![UI Drawing](https://github.com/user-attachments/assets/4f37f4d1-31c1-4534-82eb-d370fe29873a)


### ➕ Set the theme for your story
This theme is passed as part of the system message to the LLM, and helps the LLM make more a more educated decision about whether you are describing the scene to a story, or not.

### ➕ Click the Start Button
The start button will activate the listening state (Voice Activity Detection & Whisper Transcription pipelines) on the system's default input device (microphone).

### 🗣 Describe a scene to your story
Go ahead and describe a scene your story. For example, "You find yourself at the gates of a large, abandoned castle."

### 🖼️ Wait for your illustration
The scene that you just described will be passed to the LLM, which should detect it as a new scene to your story. The detailed prompt that is generated by the LLM will show up in real-time in the UI caption box, followed soon after by the illustration generated from the stable diffusion pipeline.

### 🗣 Talk about something not relevant to your story
You can test the intelligence of the LLM helper and say something not relevant to the story. For example, "Hey guys, do you think we should order a pizza?". You should find that the LLM will make the decision to disregard this, and not try to illustrate anything.

### 🪄🖼️ Interact with the animated GIF
To interact with the 3D hoverable animation created with depth maps, start an HTTP server as explained above, and you will be able to interact with the parallax.

## :bulb: Additional Tips

  If you're running on an Intel Core Ultra Series 2 laptop, and you want to set ```llm_device = 'NPU'```, be sure to have latest NPU driver installed, from [here](https://www.intel.com/content/www/us/en/download/794734/intel-npu-driver-windows.html)
  
* Based on the resolution of your display, you may want to tweak the default resolution of the illustrated image, as well as caption font size.
  To adjust the resolution of the illustrated image, look for and modify this line:
  ```
  self.image_label.setFixedSize(1216, 684)
  ```
  It's recommended to choose a 16:9 ratio resolution. You can find a convenient list [here](https://pacoup.com/2011/06/12/list-of-true-169-resolutions/).
  
  The caption font size can be adjusted by modifying this line:
  ```
  fantasy_font = QFont("Papyrus", 18, QFont.Bold)
  ```

# Additional Resources
- Learn more about [OpenVINO](https://www.intel.com/content/www/us/en/developer/tools/openvino-toolkit/overview.html)
- Explore [OpenVINO’s documentation](https://docs.openvino.ai/2023.0/home.html)

<p align="right"><a href="#top">Back to top ⬆️</a></p>