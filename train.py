#%%
import utils
import tensorflow as tf
from tensorflow.python.keras.api import keras

print(tf.__version__)

#%%

content_layers = ['block5_conv2']
style_layers = ['block1_conv1', 'block2_conv1', 'block3_conv1', 'block4_conv1', 'block5_conv1']
num_content_layers = len(content_layers)
num_style_layers = len(style_layers)

style_weight=1e-2
content_weight=1e4

#%%

#####################################
# Load the pretrained VGG19 network #
#####################################
def getVGGLayers(layer_names):
    model = keras.applications.VGG19(include_top=False, weights='imagenet')
    model.trainable = False
    print('Loaded model with name: {}'.format(model.name))
    print(model.summary())
    
    for l in model.layers:
        print(l.name)

    outputs = [model.get_layer(name).output for name in layer_names]

    # content_outputs = [model.get_layer(name).output for name in content_layers]
    # style_outputs = [model.get_layer(name).output for name in style_layers]
    # model_outputs = content_outputs + style_outputs

    # return keras.Model([model.input], content_outputs), keras.Model([model.input], style_outputs)
    return keras.Model([model.input], outputs)

#%%
#############################
# Create the loss functions #
#############################

def gram_matrix(input_tensor):
    result = tf.linalg.einsum('bijc,bijd->bcd', input_tensor, input_tensor)
    input_shape = tf.shape(input_tensor)
    num_locations = tf.cast(input_shape[1]*input_shape[2], tf.float32)
    return result/(num_locations)

#%%
##################################
# Create the Style-Content Model #
##################################

class StyleContentModel(tf.keras.models.Model):
    def __init__(self, style_layers, content_layers):
        super(StyleContentModel, self).__init__()
        self.vgg = getVGGLayers(style_layers + content_layers)
        self.style_layers = style_layers
        self.content_layers = content_layers
        self.num_style_layers = len(style_layers)
        self.vgg.trainable = False

    def call(self, inputs):
        # Expects float input in [0,1]
        inputs = inputs * 255.0
        preprocessed_input = tf.keras.applications.vgg19.preprocess_input(inputs)
        outputs = self.vgg(preprocessed_input)
        style_outputs, content_outputs = (outputs[:self.num_style_layers], outputs[self.num_style_layers:])
        style_outputs = [gram_matrix(style_output) for style_output in style_outputs]
        content_dict = {content_name:value for content_name, value in zip(self.content_layers, content_outputs)}
        style_dict = {style_name:value for style_name, value in zip(self.style_layers, style_outputs)}
        return {'content':content_dict, 'style':style_dict}

#%%
def clip_0_1(image):
  return tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=1.0)

def styleContentLoss(outputs, style_targets, content_targets):
    style_outputs = outputs['style']
    content_outputs = outputs['content']

    style_loss = tf.add_n([tf.reduce_mean((style_outputs[name]-style_targets[name])**2) 
                           for name in style_outputs.keys()])
    style_loss *= style_weight / num_style_layers

    content_loss = tf.add_n([tf.reduce_mean((content_outputs[name]-content_targets[name])**2) 
                             for name in content_outputs.keys()])
    content_loss *= content_weight / num_content_layers

    loss = style_loss + content_loss
    return loss

#%%
@tf.function()
def train_step(extractor, image, style_targets, content_targets):
    with tf.GradientTape() as tape:
        outputs = extractor(image)
        loss = styleContentLoss(outputs, style_targets, content_targets)

    grad = tape.gradient(loss, image)
    opt.apply_gradients([(grad, image)])
    image.assign(clip_0_1(image))

#%%

if __name__ == '__main__':
    print('Training neural transfer model')
    # content_image = utils.load_img('./turtle.jpg')
    content_image = utils.load_img('./feiyang.jpg')
    style_image = utils.load_img('./kandinsky.jpg')

    # utils.imshow(content_image, 'Content')
    # utils.imshow(style_image, 'Style')

    extractor = StyleContentModel(style_layers, content_layers)
    style_targets = extractor(style_image)['style']
    content_targets = extractor(content_image)['content']

    target_image = tf.Variable(content_image)

    opt = tf.optimizers.Adam(learning_rate=0.02, beta_1=0.99, epsilon=1e-1)

    epochs = 10
    steps_per_epoch = 100
    step = 0
    print('Start training')

    step_to_draw = 10

    for n in range(epochs):
        for m in range(steps_per_epoch):
            step += 1
            train_step(extractor, target_image, style_targets, content_targets)
            # print(".", end='')
            print('Epoch {}, epoch step {}, total step {}'.format(n, m, step))

            # if step % step_to_draw == 0:
            #     utils.imshow(target_image.read_value(), "Train step: {}".format(step))

        utils.imshow(target_image.read_value(), "Train step: {}".format(step))
    
    utils.imshow(target_image.read_value(), "Final")
    

        


    