import os
import subprocess
import sys
from pathlib import Path


model_dir = os.path.dirname(os.path.abspath(__file__)) + "/models"

# create the 'models' folder if it doesn't exist
Path(model_dir).mkdir(exist_ok=True)

def prepare_llm_model():
    if not Path(model_dir+"/llama-3.1-8b-instruct-awq").exists():
        print("llama Model not downloaded.")
        cmd = "optimum-cli export openvino --model meta-llama/Llama-3.1-8B-Instruct --task text-generation-with-past --weight-format int4 --group-size -1 --ratio 0.8 --awq --dataset wikitext2 --scale-estimation --sym --backup-precision int8_sym  "  + model_dir + "/llama-3.1-8b-instruct-awq/INT4_compressed_weights"
        print("llm download command:",cmd)
        os.system(cmd)
    else:
        print("llama Model already downloaded.")



def prepare_stable_diffusion_model():
    if not Path(model_dir+"/sdxl-turbo").exists():
        cmd = "optimum-cli export openvino --model stabilityai/sdxl-turbo --task stable-diffusion --weight-format fp16 " + model_dir + "/sdxl-turbo/FP16"
        os.system(cmd)
    else:
         print("Stable Diffusion SDXL Model already downloaded.")


# Define a download file helper function
def download_file(url: str, path: Path) -> None:
    """Download file."""
    import urllib.request
    path.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, path)


def prepare_voice_activity_detection_model():
    if not Path(model_dir+"/silero_vad.onnx").exists():
        print("Voice Activity Detection Model not downloaded.")
        download_file("https://github.com/snakers4/silero-vad/raw/master/src/silero_vad/data/silero_vad.onnx", Path(model_dir + "/silero_vad.onnx") )
    else:
        print("Voice Activity Detection already downloaded.")


def prepare_super_res():

    # 1032: 4x superresolution, 1033: 3x superresolution
    model_name = 'single-image-super-resolution-1033'
    base_model_dir = Path(model_dir)

    model_xml_name = f'{model_name}.xml'
    model_bin_name = f'{model_name}.bin'

    model_xml_path = base_model_dir / model_xml_name
    model_bin_path = base_model_dir / model_bin_name

    if not model_xml_path.exists():
        base_url = f'https://storage.openvinotoolkit.org/repositories/open_model_zoo/2023.0/models_bin/1/{model_name}/FP16/'
        model_xml_url = base_url + model_xml_name
        model_bin_url = base_url + model_bin_name

        download_file(model_xml_url, model_xml_path)
        download_file(model_bin_url, model_bin_path)
    else:
        print(f'{model_name} already downloaded to {base_model_dir}')


def prepare_whisper():
    whisper_version="whisper-base"
    if not Path(model_dir+"/"+whisper_version).exists():
       import huggingface_hub as hf_hub
       model_id = "OpenVINO/whisper-base-fp16-ov"
       model_path = model_dir + "/" + whisper_version

       hf_hub.snapshot_download(model_id, local_dir=model_path)
    else:
       print(f'{whisper_version} already downloaded.')


def clone_repo(repo_url: str, revision: str = None, add_to_sys_path: bool = True) -> Path:
    repo_path = Path(repo_url.split("/")[-1].replace(".git", ""))

    if not repo_path.exists():
        try:
            subprocess.run(["git", "clone", repo_url], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        except Exception as exc:
            print(f"Failed to clone the repository: {exc.stderr}")
            raise

        if revision:
            subprocess.Popen(["git", "checkout", revision], cwd=str(repo_path))
    if add_to_sys_path and str(repo_path.resolve()) not in sys.path:
        sys.path.insert(0, str(repo_path.resolve()))

    return repo_path

if __name__ == "__main__":
    prepare_llm_model()
    prepare_stable_diffusion_model()
    prepare_voice_activity_detection_model()
    prepare_super_res()
    prepare_whisper()
 


