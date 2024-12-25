import os
import sys

if "APP_PATH" in os.environ:
    app_path = os.path.abspath(os.environ["APP_PATH"])
    if os.getcwd() != app_path:
        # fix sys.path for import
        os.chdir(app_path)
    if app_path not in sys.path:
        sys.path.append(app_path)

import gradio as gr

import torch
import torchaudio
import numpy as np
import matplotlib.pyplot as plt
import re
import random
import string
from audioseal import AudioSeal


# Load generator if not already loaded in reload mode
if 'generator' not in globals():
    generator = AudioSeal.load_generator("audioseal_wm_16bits")

# Load detector if not already loaded in reload mode
if 'detector' not in globals():
    detector = AudioSeal.load_detector("audioseal_detector_16bits")


def load_audio(file):
    wav, sample_rate = torchaudio.load(file)
    return wav, sample_rate

def generate_msg_pt_by_format_string(format_string, bytes_count):
    msg_hex = format_string.replace("-", "")
    hex_length = bytes_count * 2
    binary_list = []
    for i in range(0, len(msg_hex), hex_length):
        chunk = msg_hex[i:i+hex_length]
        binary = bin(int(chunk, 16))[2:].zfill(bytes_count * 8)
        binary_list.append([int(b) for b in binary])
    # torch.randint(0, 2, (1, 16), dtype=torch.int32)
    msg_pt = torch.tensor(binary_list, dtype=torch.int32)
    return msg_pt

def embed_watermark(audio, sr, msg):
    # We add the batch dimension to the single audio to mimic the batch watermarking
    original_audio = audio.unsqueeze(0)

    watermark = generator.get_watermark(original_audio, sr, message=msg)

    watermarked_audio = original_audio + watermark

    # Alternatively, you can also call forward() function directly with different tune-down / tune-up rate
    # watermarked_audio = generator(audios, sample_rate=sr, alpha=1)

    return watermarked_audio

def generate_format_string_by_msg_pt(msg_pt, bytes_count):
    hex_length = bytes_count * 2
    binary_int = 0
    for bit in msg_pt:
        binary_int = (binary_int << 1) | int(bit.item())
    hex_string = format(binary_int, f'0{hex_length}x')

    split_hex = [hex_string[i:i + 4] for i in range(0, len(hex_string), 4)]
    format_hex = "-".join(split_hex)

    return hex_string, format_hex

def detect_watermark(audio, sr):
    # We add the batch dimension to the single audio to mimic the batch watermarking
    watermarked_audio = audio.unsqueeze(0)

    result, message = detector.detect_watermark(watermarked_audio, sr)

    # pred_prob is a tensor of size batch x 2 x frames, indicating the probability (positive and negative) of watermarking for each frame
    # A watermarked audio should have pred_prob[:, 1, :] > 0.5
    # message_prob is a tensor of size batch x 16, indicating of the probability of each bit to be 1.
    # message will be a random tensor if the detector detects no watermarking from the audio
    pred_prob, message_prob = detector(watermarked_audio, sr)

    return result, message, pred_prob, message_prob

def get_waveform_and_specgram(batch_waveform, sample_rate):
    waveform = batch_waveform.squeeze().detach().cpu().numpy()

    num_frames = waveform.shape[-1]
    time_axis = torch.arange(0, num_frames) / sample_rate

    figure, (ax1, ax2) = plt.subplots(2, 1)

    ax1.plot(time_axis, waveform, linewidth=1)
    ax1.grid(True)
    ax2.specgram(waveform, Fs=sample_rate)

    figure.suptitle(f"Waveform and specgram")

    return figure

def generate_hex_format_regex(bytes_count):
    hex_length = bytes_count * 2
    hex_string = 'F' * hex_length
    split_hex = [hex_string[i:i + 4] for i in range(0, len(hex_string), 4)]
    format_like = "-".join(split_hex)
    regex_pattern = '^' + '-'.join([r'[0-9A-Fa-f]{4}'] * len(split_hex)) + '$'
    return format_like, regex_pattern

def generate_hex_random_message(bytes_count):
    hex_length = bytes_count * 2
    hex_string = ''.join(random.choice(string.hexdigits) for _ in range(hex_length))
    split_hex = [hex_string[i:i + 4] for i in range(0, len(hex_string), 4)]
    random_str = "-".join(split_hex)
    return random_str, "".join(split_hex)

with gr.Blocks(title="AudioSeal") as demo:
    gr.Markdown("""
    # AudioSeal Demo

    Find the project [here](https://github.com/facebookresearch/audioseal.git).
    """)

    with gr.Tabs():
        with gr.TabItem("Embed Watermark"):
            with gr.Row():
                with gr.Column():
                    embedding_aud = gr.Audio(label="Input Audio", type="filepath")
                    embedding_specgram = gr.Checkbox(label="Show specgram", value=False, info="Show debug information")

                    embedding_type = gr.Radio(["random", "input"], value="random", label="Type", info="Type of watermarks")

                    nbytes = int(generator.msg_processor.nbits / 8)
                    format_like, regex_pattern = generate_hex_format_regex(nbytes)
                    msg, _ = generate_hex_random_message(nbytes)
                    embedding_msg = gr.Textbox(
                        label=f"Message ({nbytes} bytes hex string)",
                        info=f"format like {format_like}",
                        value=msg,
                        interactive=False, show_copy_button=True)

                    embedding_btn = gr.Button("Embed Watermark")
                with gr.Column():
                    marked_aud = gr.Audio(label="Output Audio", show_download_button=True)
                    specgram_original = gr.Plot(label="Original Audio", format="png", visible=False)
                    specgram_watermarked = gr.Plot(label="Watermarked Audio", format="png", visible=False)


            def change_embedding_type(type):
                if type == "random":
                    msg, _ = generate_hex_random_message(nbytes)
                    return gr.update(interactive=False, value=msg)
                else:
                    return gr.update(interactive=True)
            embedding_type.change(
                fn=change_embedding_type,
                inputs=[embedding_type],
                outputs=[embedding_msg]
            )

            def check_embedding_msg(msg):
                if not re.match(regex_pattern, msg):
                    gr.Warning(
                        f"Invalid format. Please use like '{format_like}'",
                        duration=0)
            embedding_msg.change(
                fn=check_embedding_msg,
                inputs=[embedding_msg],
                outputs=[]
            )

            def run_embed_watermark(file, show_specgram, type, msg):
                if file is None:
                    raise gr.Erro("No file uploaded", duration=5)
                if not re.match(regex_pattern, msg):
                    raise gr.Error(f"Invalid format. Please use like '{format_like}'", duration=5)

                audio_original, rate = load_audio(file)
                msg_pt = generate_msg_pt_by_format_string(msg, nbytes)
                audio_watermarked = embed_watermark(audio_original, rate, msg_pt)
                output = rate, audio_watermarked.squeeze().detach().cpu().numpy().astype(np.float32)

                if show_specgram:
                    fig_original = get_waveform_and_specgram(audio_original, rate)
                    fig_watermarked = get_waveform_and_specgram(audio_watermarked, rate)
                    return [
                        output,
                        gr.update(visible=True, value=fig_original),
                        gr.update(visible=True, value=fig_watermarked)]
                else:
                    return [
                        output,
                        gr.update(visible=False),
                        gr.update(visible=False)]

            embedding_btn.click(
                fn=run_embed_watermark,
                inputs=[embedding_aud, embedding_specgram, embedding_type, embedding_msg],
                outputs=[marked_aud, specgram_original, specgram_watermarked]
            )

        with gr.TabItem("Detect Watermark"):
            with gr.Row():
                with gr.Column():
                    detecting_aud = gr.Audio(label="Input Audio", type="filepath")
                with gr.Column():
                    detecting_btn = gr.Button("Detect Watermark")
                    predicted_messages = gr.JSON(label="Detected Messages")

            def run_detect_watermark(file):
                if file is None:
                    raise gr.Error("No file uploaded", duration=5)

                audio_watermarked, rate = load_audio(file)
                result, message, pred_prob, message_prob = detect_watermark(audio_watermarked, rate)

                _, fromat_msg = generate_format_string_by_msg_pt(message[0], nbytes)

                sum_above_05 = (pred_prob[:, 1, :] > 0.5).sum(dim=1)

                # Create message output as JSON
                message_json = {
                    "socre": result,
                    "message": fromat_msg,
                    "frames_count_all": pred_prob.shape[2],
                    "frames_count_above_05": sum_above_05[0].item(),
                    "bits_probability": message_prob[0].tolist(),
                    "bits_massage": message[0].tolist()
                }
                return message_json
            detecting_btn.click(
                fn=run_detect_watermark,
                inputs=[detecting_aud],
                outputs=[predicted_messages]
            )

if __name__ == "__main__":
    demo.launch()
