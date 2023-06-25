
// URL of image to use. Change this to your image.
const IMAGE_URL = 'https://samples.clarifai.com/metro-north.jpg';

const raw = JSON.stringify({
  "user_app_id": {
    "user_id": "datastrategy",
    "app_id": "MAIN-image-moderation"
  },
  "inputs": [
      {
          "data": {
              "image": {
                  "url": IMAGE_URL
              }
          }
      }
  ]
});

const requestOptions = {
    method: 'POST',
    headers: {
        'Accept': 'application/json',
        'Authorization': 'Key ' + 'YOUR_PAT_HERE'
    },
    body: raw
};

// NOTE: MODEL_VERSION_ID is optional, you can also call prediction with the MODEL_ID only
// https://api.clarifai.com/v2/models/{YOUR_MODEL_ID}/outputs
// this will default to the latest version_id

fetch(`https://api.clarifai.com/v2/models/moderation-all-resnext-2/versions/178f26f97dd648368869cdab09f3d9aa/outputs`, requestOptions)
    .then(response => response.text())
    .then(result => console.log(result))
    .catch(error => console.log('error', error));
  