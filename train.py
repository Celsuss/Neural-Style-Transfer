#%%
import os
import sys
import time
import utils
import tensorflow as tf
from tensorflow.python.keras.api import keras

print(tf.__version__)

#%%

DRAW = False
image_base_path = 'generated'

content_layers = ['block5_conv2']
style_layers = ['block1_conv1', 'block2_conv1', 'block3_conv1', 'block4_conv1', 'block5_conv1']
num_content_layers = len(content_layers)
num_style_layers = len(style_layers)

style_weight=1e-2
content_weight=1e4
total_variation_weight=1e8

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

########################
# Total variation loss #
########################
def high_pass_x_y(image):
    x_var = image[:,:,1:,:] - image[:,:,:-1,:]
    y_var = image[:,1:,:,:] - image[:,:-1,:,:]

    return x_var, y_var

def total_variation_loss(image):
    x_deltas, y_deltas = high_pass_x_y(image)
    return tf.reduce_mean(x_deltas**2) + tf.reduce_mean(y_deltas**2)


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
def train_step(extractor, image, style_targets, content_targets, optimizer):
    with tf.GradientTape() as tape:
        outputs = extractor(image)
        loss = styleContentLoss(outputs, style_targets, content_targets)
        loss += total_variation_weight*total_variation_loss(image)

    grad = tape.gradient(loss, image)
    optimizer.apply_gradients([(grad, image)])
    image.assign(clip_0_1(image))
    return loss

#%%
#####################
# Train the network #
#####################
def train():
    print('Training neural transfer model')
    # content_image = utils.load_img('./turtle.jpg')
    content_image = utils.load_img('./feiyang.jpg')
    style_image = utils.load_img('./van_gogh.jpeg')

    # utils.imshow(content_image, 'Content')
    # utils.imshow(style_image, 'Style')

    extractor = StyleContentModel(style_layers, content_layers)
    style_targets = extractor(style_image)['style']
    content_targets = extractor(content_image)['content']

    target_image = tf.Variable(content_image)

    optimizer = tf.optimizers.Adam(learning_rate=0.02, beta_1=0.99, epsilon=1e-1)

    epochs = 10
    steps_per_epoch = 100
    step = 0
    print('Start training')

    step_to_draw = 10
    step_to_save_image = 100

    start_time = time.time()
    image_name = 'feiyang_van_gogh'
    image_path = os.path.join(image_base_path, '{}_{}'.format(image_name, time.strftime("%Y-%m-%d-%H%M")))

    utils.save_img(target_image, image_path, '{}_original'.format(image_name))

    best_loss = 9000000000
    best_image = None

    for n in range(epochs):
        for m in range(steps_per_epoch):
            step += 1
            loss = train_step(extractor, target_image, style_targets, content_targets, optimizer)
            # print(".", end='')
            print('Epoch {}, epoch step {}, total step {}, loss: {}'.format(n, m, step, loss))

            if loss < best_loss:
                best_loss = loss
                best_image = target_image

            if step % step_to_save_image == 0:
                utils.save_img(target_image, image_path, '{}_step_{}'.format(image_name, step))


    utils.save_img(best_image, image_path, '{}_final'.format(image_name))


#%%
def handleArguments():
    global DRAW
    if len(sys.argv) > 1:
        for arg in sys.argv[1:]:
            if arg == '-d':
                DRAW = True

if __name__ == '__main__':
    handleArguments()
    train()
    

        


    