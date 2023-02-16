import tensorflow as tf

class GAN(tf.keras.Model):
    def __init__(self, discriminator, generator):
        super(GAN, self).__init__()
        self.discriminator = discriminator
        self.generator = generator
        self.gloss_tracker = tf.keras.metrics.Mean(name="g_loss")
        self.dloss_tracker = tf.keras.metrics.Mean(name="d_loss")
        self.mse_metric = tf.keras.metrics.MeanSquaredError(name="mse")
        self.mae_metric = tf.keras.metrics.MeanAbsoluteError(name="mae")
        self.val_gloss_tracker = tf.keras.metrics.Mean(name="g_loss")
        self.val_dloss_tracker = tf.keras.metrics.Mean(name="d_loss")
        self.val_mse_metric = tf.keras.metrics.MeanSquaredError(name="mse")
        self.val_mae_metric = tf.keras.metrics.MeanAbsoluteError(name="mae")


    def compile(self, d_optimizer, g_optimizer, loss_fn):
        super(GAN, self).compile()
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        self.loss_fn = loss_fn
        self.mse = tf.keras.losses.MeanSquaredError()

    def call(self, dataset):
        gen_output = self.generator(dataset)
        disc_output = self.discriminator([dataset, gen_output], training=False)
        
        return gen_output#, disc_output
        
    def save(self,path,overwrite):
        tf.keras.models.save_model(self.discriminator,'./model_chk/discriminator_' + path, overwrite)
        tf.keras.models.save_model(self.generator,'./model_chk/generator_' + path, overwrite)
        
#     def reduce_lr(self):
#         print("val loss failed to improve 10 validations in a row")
#         print("Current Generator LR: " + str(self.g_optimizer.lr) + "reducing learning rate by 10%")
#         tf.keras.backend.set_value(self.g_optimizer.lr, self.g_optimizer.lr * .90)
#         print("New Generator LR: " + str(tf.keras.backend.get_value(self.g_optimizer.lr)))
#         print("Current Discriminator LR: " + str(self.d_optimizer.lr) + "reducing learning rate by 10%")
#         tf.keras.backend.set_value(self.d_optimizer.lr, self.d_optimizer.lr * .90)
#         print("New Discriminator LR: " + str(tf.keras.backend.get_value(self.d_optimizer.lr))) 
    def generator_loss(self, disc_generated_output, gen_output, target):
        gan_loss = self.loss_fn(tf.ones_like(disc_generated_output), disc_generated_output)

        # Mean absolute error
        l1_loss = tf.reduce_mean(tf.abs(target - gen_output))

        total_gen_loss = gan_loss + (100 * l1_loss)

        return total_gen_loss, gan_loss, l1_loss

    def discriminator_loss(self, disc_real_output, disc_generated_output):
        real_loss = self.loss_fn(tf.ones_like(disc_real_output), disc_real_output)
        
        generated_loss = self.loss_fn(tf.zeros_like(disc_generated_output), disc_generated_output)
        
        total_disc_loss = real_loss + generated_loss

        return total_disc_loss
        
    def train_step(self, dataset):
        
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            
            input_dataset, target_dataset = dataset

            gen_output = self.generator(input_dataset, training=True)

            disc_real_output = self.discriminator([input_dataset, target_dataset], training=True)
            disc_generated_output = self.discriminator([input_dataset, gen_output], training=True)

            gen_total_loss, gen_gan_loss, gen_l1_loss = self.generator_loss(disc_generated_output, gen_output, target_dataset)
            disc_loss = self.discriminator_loss(disc_real_output, disc_generated_output)

        generator_gradients = gen_tape.gradient(gen_total_loss,
                                          self.generator.trainable_variables)
        discriminator_gradients = disc_tape.gradient(disc_loss,
                                               self.discriminator.trainable_variables)

        self.g_optimizer.apply_gradients(zip(generator_gradients,
                                          self.generator.trainable_variables))
        self.d_optimizer.apply_gradients(zip(discriminator_gradients,
                                              self.discriminator.trainable_variables))
        
        
        self.dloss_tracker.update_state(disc_loss)
         
        self.gloss_tracker.update_state(gen_total_loss)

        self.mae_metric.update_state(target_dataset, tf.squeeze(gen_output))
        self.mse_metric.update_state(target_dataset, tf.squeeze(gen_output))
        
        return {"d_loss": self.dloss_tracker.result(), "g_loss": self.gloss_tracker.result(), "mae": self.mae_metric.result(), "mse": self.mse_metric.result()}
    
    def test_step(self, dataset):
                
        input_dataset, target_dataset = dataset

        gen_output = self.generator(input_dataset, training=False)

        disc_real_output = self.discriminator([input_dataset, target_dataset], training=False)
        disc_generated_output = self.discriminator([input_dataset, gen_output], training=False)

        gen_total_loss, gen_gan_loss, gen_l1_loss = self.generator_loss(disc_generated_output, gen_output, target_dataset)
        disc_loss = self.discriminator_loss(disc_real_output, disc_generated_output)

        self.val_dloss_tracker.update_state(disc_loss)

        self.val_gloss_tracker.update_state(gen_total_loss)

        self.val_mae_metric.update_state(target_dataset, tf.squeeze(gen_output))
        self.val_mse_metric.update_state(target_dataset, tf.squeeze(gen_output))
        
        return {"d_loss": self.val_dloss_tracker.result(), "g_loss": self.val_gloss_tracker.result(), "mae": self.val_mae_metric.result(), "mse": self.val_mse_metric.result()}

    @property
    def metrics(self):
        # We list our `Metric` objects here so that `reset_states()` can be
        # called automatically at the start of each epoch
        # or at the start of `evaluate()`.
        # If you don't implement this property, you have to call
        # `reset_states()` yourself at the time of your choosing.
        return [self.dloss_tracker, self.gloss_tracker, self.mae_metric, self.mse_metric,
               self.val_dloss_tracker, self.val_gloss_tracker, self.val_mae_metric, self.val_mse_metric]