from tensorflow.keras.layers import Input, Activation, Dense,Flatten, BatchNormalization, Add, Conv2D,GaussianNoise
from tensorflow.keras.layers import MaxPooling2D,AveragePooling2D,Permute,Reshape,LSTM,Lambda,GRU,Bidirectional,BatchNormalization,Concatenate,GlobalAveragePooling1D
from tensorflow.keras import regularizers
from tensorflow.keras.optimizers import Adam
from models.attention_layer import *
from tensorflow.keras.models import Model
from utils import loss_with_constraint,sharpe_ratio_loss,sharpe_ratio
import tensorflow.keras.backend as K
from models.norm_layer import normalize_layer
###############################
# additive attention RNN models
###############################
def build_self_att_add_att_model(params):
    units = params['units']
    activation = params['activation']
    reg1 = params['l2']
    reg2 = params['l2_1']
    lr = params['l2_2']
    input_shape = params['input_shape']
    type_norm = params['type_norm']

    ts = input_shape[1]
    tickers = input_shape[0]

    input = Input(shape=input_shape)
    reshape_inp = Lambda(lambda x: K.permute_dimensions(x,pattern=(0,2,1,3))) (input)
    reshape_inp = Reshape((ts,-1)) (reshape_inp)

    batch_norm = normalize_layer(type_norm)(reshape_inp)


    prob = SelfAttentionLayer(latent_dim=32,name='Self_Att',kernel_regularizer=regularizers.l2(1e-4))(batch_norm)
    att = Lambda(lambda x: K.batch_dot(x[0],x[1]) ) ([prob,batch_norm])

    # batch_norm_2 = BatchNormalization()(recurrent_layer)
    out = GlobalAveragePooling1D()(att)

    ##ATTENTION LAYER
    # contxt_layer = AdditiveAttentionLayer(name='Att',latent_dim=32,kernel_regularizer=regularizers.l2(0.01))([batch_norm,batch_norm_2])

    contxt_layer = AdditiveAttentionLayer(name='Add_Att',latent_dim=32,kernel_regularizer=regularizers.l2(0.01))([batch_norm,out])
    merge = Concatenate()([out,contxt_layer])


    out = Dense(units, kernel_regularizer =regularizers.l2(reg2),activation='tanh') (merge)
    batch_norm_3 = normalize_layer(type_norm)(out)


    out = Dense(tickers, kernel_regularizer =regularizers.l2(reg2)) (batch_norm_3)

    out = Activation('sigmoid')(out)

    model = Model([input], [out])
    optimizer = Adam(lr = lr)
    model.compile(loss=sharpe_ratio_loss, optimizer=optimizer, metrics = [sharpe_ratio])
    return model


######################################

### 2_layer Self Attention

######################################
def build_self_att_self_att_model(params):
    units = params['units']
    activation = params['activation']
    reg1 = params['l2']
    reg2 = params['l2_1']
    lr = params['l2_2']
    input_shape = params['input_shape']
    type_norm = params['type_norm']

    ts = input_shape[1]
    tickers = input_shape[0]

    input = Input(shape=input_shape)
    reshape_inp = Lambda(lambda x: K.permute_dimensions(x,pattern=(0,2,1,3))) (input)
    reshape_inp = Reshape((ts,-1)) (reshape_inp)

    batch_norm = normalize_layer(type_norm)(reshape_inp)

    ## layer 1
    prob = SelfAttentionLayer(latent_dim=32,name='Self_Att',kernel_regularizer=regularizers.l2(1e-4))(batch_norm)
    att = Lambda(lambda x: K.batch_dot(x[0],x[1]) ) ([prob,batch_norm])
    
    batch_norm_2 = normalize_layer(type_norm)(att)


    ## layer 2
    prob_2 = SelfAttentionLayer(latent_dim=32,name='Self_Att_2',kernel_regularizer=regularizers.l2(1e-4))(batch_norm_2)
    att_2 = Lambda(lambda x: K.batch_dot(x[0],x[1]) ) ([prob_2,batch_norm_2])

    out = GlobalAveragePooling1D()(att_2)
    out = Dense(units, kernel_regularizer =regularizers.l2(reg2),activation='tanh') (out)
    batch_norm_3 = normalize_layer(type_norm)(out)


    out = Dense(tickers,kernel_regularizer =regularizers.l2(reg2)) (batch_norm_3)

    out = Activation('sigmoid')(out)

    model = Model([input], [out])
    optimizer = Adam(lr = lr)
    model.compile(loss=sharpe_ratio_loss, optimizer=optimizer, metrics = [sharpe_ratio])
    return model

##############################
### 2_SA_AA
##############################

def build_2_SA_AA_model(params):
    units = params['units']
    activation = params['activation']
    reg1 = params['l2']
    reg2 = params['l2_1']
    lr = params['l2_2']
    input_shape = params['input_shape']
    type_norm = params['type_norm']

    ts = input_shape[1]
    tickers = input_shape[0]

    input = Input(shape=input_shape)
    reshape_inp = Lambda(lambda x: K.permute_dimensions(x,pattern=(0,2,1,3))) (input)
    reshape_inp = Reshape((ts,-1)) (reshape_inp)

    batch_norm = normalize_layer(type_norm)(reshape_inp)

    ## layer 1
    prob = SelfAttentionLayer(latent_dim=32,name='Self_Att',kernel_regularizer=regularizers.l2(1e-4))(batch_norm)
    att = Lambda(lambda x: K.batch_dot(x[0],x[1]) ) ([prob,batch_norm])

    batch_norm_3 = normalize_layer(type_norm)(att)

    ## layer 2
    prob_2 = SelfAttentionLayer(latent_dim=32,name='Self_Att_2',kernel_regularizer=regularizers.l2(1e-4))(batch_norm_3)
    att_2 = Lambda(lambda x: K.batch_dot(x[0],x[1]) ) ([prob_2,batch_norm_3])

    out = GlobalAveragePooling1D()(att_2)

    ## AA layer
    contxt_layer = AdditiveAttentionLayer(name='Add_Att',latent_dim=32,kernel_regularizer=regularizers.l2(0.01))([batch_norm,out])
    merge = Concatenate()([out,contxt_layer])


    out = Dense(units, kernel_regularizer =regularizers.l2(reg2),activation='tanh') (merge)
    batch_norm_4 = normalize_layer(type_norm)(out)


    out = Dense(tickers,kernel_regularizer =regularizers.l2(reg2)) (batch_norm_4)

    out = Activation('sigmoid')(out)

    model = Model([input], [out])
    optimizer = Adam(lr = lr)
    model.compile(loss=sharpe_ratio_loss, optimizer=optimizer, metrics = [sharpe_ratio])
    return model



###############################
# additive attention RNN models
###############################
def build_vae_add_att_lstm_model(params):
    units = params['units']
    activation = params['activation']
    reg1 = params['l2']
    reg2 = params['l2_1']
    lr = params['l2_2']
    input_shape = params['input_shape']
    type_norm = params['type_norm']

    ts = input_shape[1]
    tickers = input_shape[0]

    input = Input(shape=input_shape)
    reshape_inp = Lambda(lambda x: K.permute_dimensions(x,pattern=(0,2,1,3))) (input)
    reshape_inp = Reshape((ts,-1)) (reshape_inp)

    batch_norm = normalize_layer(type_norm)(reshape_inp)


    recurrent_layer = LSTM(units = units,
                    activation = activation,
                  kernel_regularizer=regularizers.l2(reg1)) (batch_norm)

    batch_norm_2 = normalize_layer(type_norm)(recurrent_layer)


    ##ATTENTION LAYER
    contxt_layer = AdditiveAttentionLayer(name='Att',latent_dim=32,kernel_regularizer=regularizers.l2(1e-5))([batch_norm,batch_norm_2])

    merge = Concatenate()([batch_norm_2,contxt_layer])

    mean = Dense(16, kernel_regularizer =regularizers.l2(1e-5)) (merge)
    std = Dense(16, kernel_regularizer =regularizers.l2(1e-5),activation='relu') (merge)

    ##slice tensor
    slice = Lambda(lambda x: x[:,0:1]) (mean)
    ## random tensor for reparameterizing

    ran = GaussianNoise(stddev=1)(slice)
    out = Lambda(lambda x: x[0] * x[1] + x[2]) ([ran,std,mean])


    out = Dense(units, kernel_regularizer =regularizers.l2(1e-5)) (out)
    batch_norm_3 = BatchNormalization()(out)

    out = Dense(64, kernel_regularizer =regularizers.l2(1e-5)) (batch_norm_3)
    batch_norm_4 = BatchNormalization()(out)

    out = Dense(128, kernel_regularizer =regularizers.l2(1e-5)) (batch_norm_4)
    batch_norm_5 = normalize_layer(type_norm)(out)

    out = Dense(256, kernel_regularizer =regularizers.l2(1e-5)) (batch_norm_5)
    batch_norm_6 = normalize_layer(type_norm)(out)

    out = Dense(tickers, kernel_regularizer =regularizers.l2(1e-5)) (batch_norm_6)

    out = Activation('sigmoid')(out)

    # # KL post/prior
    # post = Dense(256, kernel_regularizer =regularizers.l2(1e-5)) (out)
    # batch_norm_7 = BatchNormalization()(post)
    #
    # post = Dense(128, kernel_regularizer =regularizers.l2(1e-5)) (batch_norm_7)
    # batch_norm_8 = BatchNormalization()(post)
    #
    # post = Dense(64, kernel_regularizer =regularizers.l2(1e-5)) (batch_norm_8)
    # post = BatchNormalization()(post)
    #
    # ## mean and std of posterior
    # mean1 = Dense(16, kernel_regularizer =regularizers.l2(1e-5)) (post)
    # std1 = Dense(16, kernel_regularizer =regularizers.l2(1e-5),activation='relu') (post)

    model = Model([input], [out])
    optimizer = Adam(lr = lr)
    loss_func = loss_with_constraint( mean, K.pow(std,2))
    model.compile(loss=loss_func, optimizer=optimizer, metrics = [sharpe_ratio])
    return model
