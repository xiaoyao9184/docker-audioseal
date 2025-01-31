"""
This file contains tests for the API of your model. You can run these tests by installing test requirements:

    ```bash
    pip install -r requirements-test.txt
    ```
Then execute `pytest` in the directory of this file.

- Change `NewModel` to the name of the class in your model.py file.
- Change the `request` and `expected_response` variables to match the input and output of your model.
"""
import os.path

import pytest
import json
from model import AudioSeal
import responses


@pytest.fixture
def client():
    from _wsgi import init_app
    app = init_app(model_class=AudioSeal)
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client


@pytest.fixture
def model_dir_env(tmp_path, monkeypatch):
    model_dir = tmp_path / "model_dir"
    model_dir.mkdir()
    monkeypatch.setattr(AudioSeal, 'MODEL_DIR', str(model_dir))
    return model_dir


@responses.activate
def test_predict(client, model_dir_env):
    responses.add(
        responses.GET,
        'http://test_predict.wam.ml-backend.com/audio.wav',
        body=open(os.path.join(os.path.dirname(__file__), 'test_audio', 'audio.wav'), 'rb').read(),
        status=200
    )
    request = {
        'tasks': [{
            'data': {
                'audio': 'http://test_predict.wam.ml-backend.com/audio.wav'
            }
        }],
        # Your labeling configuration here
        'label_config': '''
        <View>
  <Audio name="audio" value="$audio" zoom="true" hotkey="ctrl+enter" />

  <Header value="watermark label:"/>
  <Labels name="watermark_mask" toName="audio">
    <Label value="watermarked" />
  </Labels>

  <TextArea name="watermark_msg" toName="audio"
            maxSubmissions="1"
            editable="false"
            displayMode="region-list"
            rows="1"
            required="true"
            perRegion="true"
            />
</View>
'''
    }

    response = client.post('/predict', data=json.dumps(request), content_type='application/json')
    assert response.status_code == 200
    response = json.loads(response.data)
    expected_texts = {
        '1984'
    }
    texts_response = set()
    for r in response['results'][0]['result']:
        if r['from_name'] == 'watermark_msg':
            texts_response.add(r['value']['text'][0])
    assert texts_response == expected_texts
