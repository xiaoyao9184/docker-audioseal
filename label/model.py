import os
import sys
import logging
import torch
import torchaudio
from uuid import uuid4

from audioseal import AudioSeal

from typing import List, Dict, Optional
from label_studio_ml.model import LabelStudioMLBase
from label_studio_ml.response import ModelResponse
from label_studio_ml.utils import DATA_UNDEFINED_NAME
from label_studio_sdk._extensions.label_studio_tools.core.utils.io import get_local_path

logger = logging.getLogger(__name__)


# Device configuration
device = "cuda" if torch.cuda.is_available() else "cpu"

detector = AudioSeal.load_detector("audioseal_detector_16bits")
detector = detector.to(device)
detector_nbytes = int(detector.nbits / 8)

class AudioSeal(LabelStudioMLBase):
    """Custom ML Backend model
    """
    # score threshold to wipe out noisy results
    SCORE_THRESHOLD = float(os.getenv('SCORE_THRESHOLD', 0.3))

    # Label Studio image upload folder:
    # should be used only in case you use direct file upload into Label Studio instead of URLs
    LABEL_STUDIO_ACCESS_TOKEN = (
        os.environ.get("LABEL_STUDIO_ACCESS_TOKEN") or os.environ.get("LABEL_STUDIO_API_KEY")
    )
    LABEL_STUDIO_HOST = (
        os.environ.get("LABEL_STUDIO_HOST") or os.environ.get("LABEL_STUDIO_URL")
    )

    MODEL_DIR = os.getenv('MODEL_DIR', '.')

    def setup(self):
        """Configure any paramaters of your model here
        """
        self.set("model_version", f'{self.__class__.__name__}-v0.0.1')

    # def _get_image_url(self, task, value):
    #     # TODO: warning! currently only s3 presigned urls are supported with the default keys
    #     # also it seems not be compatible with file directly uploaded to Label Studio
    #     # check RND-2 for more details and fix it later
    #     image_url = task['data'].get(value) or task['data'].get(DATA_UNDEFINED_NAME)

    #     if image_url.startswith('s3://'):
    #         # presign s3 url
    #         r = urlparse(image_url, allow_fragments=False)
    #         bucket_name = r.netloc
    #         key = r.path.lstrip('/')
    #         client = boto3.client('s3')
    #         try:
    #             image_url = client.generate_presigned_url(
    #                 ClientMethod='get_object',
    #                 Params={'Bucket': bucket_name, 'Key': key}
    #             )
    #         except ClientError as exc:
    #             logger.warning(f'Can\'t generate presigned URL for {image_url}. Reason: {exc}')
    #     return image_url

    def load_audio(self, file):
        wav, sample_rate = torchaudio.load(file)
        return wav, sample_rate

    def detect_watermark(self, audio, sr):
        watermarked_audio = audio.to(device)

        # If the audio has more than one channel, average all channels to 1 channel
        if watermarked_audio.shape[0] > 1:
            mono_audio = torch.mean(watermarked_audio, dim=0, keepdim=True)
        else:
            mono_audio = watermarked_audio

        # We add the batch dimension to the single audio to mimic the batch watermarking
        batched_audio = mono_audio.unsqueeze(0)

        result, message = detector.detect_watermark(batched_audio, sr)

        # pred_prob is a tensor of size batch x 2 x frames, indicating the probability (positive and negative) of watermarking for each frame
        # A watermarked audio should have pred_prob[:, 1, :] > 0.5
        # message_prob is a tensor of size batch x 16, indicating of the probability of each bit to be 1.
        # message will be a random tensor if the detector detects no watermarking from the audio
        pred_prob, message_prob = detector(batched_audio, sr)

        return result, message, pred_prob, message_prob

    def generate_format_string_by_msg_pt(self, msg_pt, bytes_count):
        hex_length = bytes_count * 2
        binary_int = 0
        for bit in msg_pt:
            binary_int = (binary_int << 1) | int(bit.item())
        hex_string = format(binary_int, f'0{hex_length}x')

        split_hex = [hex_string[i:i + 4] for i in range(0, len(hex_string), 4)]
        format_hex = "-".join(split_hex)

        return hex_string, format_hex

    def predict_single(self, task):
        logger.debug('Task data: %s', task['data'])
        from_name_area, to_name, value = self.get_first_tag_occurence('TextArea', 'Audio', name_filter=lambda x: x == 'watermark_msg')
        from_name_label, _, _ = self.get_first_tag_occurence('Labels', 'Audio', name_filter=lambda x: x == 'watermark_mask')

        labels = []
        for idx, l in enumerate(self.label_interface.labels):
            if 'watermarked' in l.keys() and l['watermarked'].parent_name == 'watermark_mask':
                labels.append(l)
        if len(labels) != 1:
            logger.error("More than one 'watermarked' label in the tag.")
        label = 'watermarked'

        id_gen = str(uuid4())[:4]

        audio_url = task['data'].get(value) or task['data'].get(DATA_UNDEFINED_NAME)
        audio_path = get_local_path(audio_url, task_id=task.get('id'))

        # run detect
        audio_watermarked, rate = self.load_audio(audio_path)
        audio_watermarked_duration = audio_watermarked.shape[1] / rate
        score, message, pred_prob, message_prob = self.detect_watermark(audio_watermarked, rate)
        _, fromat_msg = self.generate_format_string_by_msg_pt(message[0], detector_nbytes)

        result = []
        if score < self.SCORE_THRESHOLD:
            logger.info(f'Skipping result with low score: {score}')
        else:
            # must add one for the label
            result.append({
                'id': id_gen,
                'from_name': from_name_label,
                'to_name': to_name,
                'type': 'labels',
                'original_length': audio_watermarked_duration,
                'value': {
                    'labels': [label],
                    "start": 0,
                    "end": audio_watermarked_duration,
                    "channel": 1,
                    "text": [fromat_msg]
                },
                'score': score
            })
            # and one for the TextArea
            result.append({
                'id': id_gen,
                'from_name': from_name_area,
                'to_name': to_name,
                'type': 'textarea',
                'value': {
                    "text": [fromat_msg]
                },
                'score': score
            })
        return {
            'result': result,
            'score': score,
            'model_version': self.get('model_version'),
        }

    def predict(self, tasks: List[Dict], context: Optional[Dict] = None, **kwargs) -> ModelResponse:
        predictions = []
        for task in tasks:
            # TODO: implement is_skipped() function
            # if is_skipped(task):
            #     continue

            prediction = self.predict_single(task)
            if prediction:
                predictions.append(prediction)

        return ModelResponse(predictions=predictions, model_versions=self.get('model_version'))
