import tensorflow as tf



class NALU:
	def __init__(self,input_shape=(0,0),size=2,epsilon = 1e-8,name = ""):
		print("hello NALU world "+name)
		

		self.size = size # the number of neurons or units in the NALU
		self.input_shape = input_shape # tuple describing the shape of the input to the NALU (observations,columns)
		self.epsilon = epsilon #used to avoid log of 0 
		self.name = name

		weights_shape = (self.input_shape[1],size)

		with tf.name_scope(name):
			self.W_hat = tf.get_variable(name+"W_hat",shape = weights_shape)
			self.M_hat = tf.get_variable(name+"M_hat",shape = weights_shape)
			self.G = tf.get_variable(name+"G",shape = weights_shape)



	def NALU_output(self,X):
		# NAC: a = Wx W = tanh(Wˆ ) * σ(Mˆ )

		
		W = tf.nn.tanh(self.W_hat) *  tf.nn.sigmoid(self.M_hat)
		a = tf.matmul(X,W)

		# NALU: y = g * a + (1 − g) *m  m = expW(log(|x| + epsilon)), g = σ(Gx)
		g = tf.nn.sigmoid(tf.matmul(X,self.G))
		m = tf.exp(tf.matmul(tf.log(tf.abs(X) + self.epsilon),W))

		y = (g*a) + (1-g)*m

		return y