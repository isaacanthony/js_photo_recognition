"use strict";

var ndarray = require('ndarray');
var ops = require('ndarray-ops');

(function() {
  var model = new KerasJS.Model({
    filepaths: {
      model: 'https://transcranial.github.io/keras-js-demos-data/inception_v3/inception_v3.json',
      weights: 'https://transcranial.github.io/keras-js-demos-data/inception_v3/inception_v3_weights.buf',
      metadata: 'https://transcranial.github.io/keras-js-demos-data/inception_v3/inception_v3_metadata.json'
    },
    gpu: true
  });

  var topClasses = function(predictions) {
    return Array.prototype.slice.call(predictions)
             .map(function(prediction, index) { return { 'probability': prediction, 'name': imagenetClasses[index + ''][1] }; })
             .sort(function(a, b) { return b.probability - a.probability; });
  };

  var recognize = function(imageData) {
    model.ready()
    .then(function() {
      var { data, width, height } = imageData;

      var dataTensor = ndarray(new Float32Array(data), [width, height, 4]);
      var dataProcessedTensor = ndarray(new Float32Array(width * height * 3), [width, height, 3]);
      ops.divseq(dataTensor, 255);
      ops.subseq(dataTensor, 0.5);
      ops.mulseq(dataTensor, 2);
      ops.assign(dataProcessedTensor.pick(null, null, 0), dataTensor.pick(null, null, 0));
      ops.assign(dataProcessedTensor.pick(null, null, 1), dataTensor.pick(null, null, 1));
      ops.assign(dataProcessedTensor.pick(null, null, 2), dataTensor.pick(null, null, 2));

      return model.predict({ input_1: dataProcessedTensor.data });
    })
    .then(function(data) {
      var text = topClasses(data.predictions)
                   .filter(function(obj) { return obj.probability > 0.3; })
                   .map(function(obj) { return obj.name.replace('_', ' '); })
                   .join('<br>');
      document.querySelector('.output').innerHTML = text;
    })
    .catch(function(err) {
      console.log(err);
    });
  };

  var process = function(photo) {
    document.querySelector('.thumbnail').src = photo;
    var img = new Image();
    img.onload = function() {
      var canvas = document.querySelector('.canvas');
      canvas.width = 299;
      canvas.height = 299;

      var ctx = canvas.getContext('2d');
      ctx.drawImage(img, 0, 0, img.width, img.height, 0, 0, canvas.width, canvas.height);
      recognize(ctx.getImageData(0, 0, canvas.width, canvas.height));
    };
    img.src = photo;
  };

  document.querySelector('.upload').onchange = function(evt) {
    var reader = new FileReader();
    reader.onload = function() { process(reader.result); };
    reader.readAsDataURL(evt.target.files[0]);
  };

  document.querySelector('.thumbnail').addEventListener('click', function() {
    $('.upload').click();
  });
})();
