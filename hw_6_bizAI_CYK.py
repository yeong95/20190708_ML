# -*- coding: utf-8 -*-

# =============================================================================
# # BIZ&AI lab 6기 과제_6
# 
# 
# * 리뷰글 긍부정 예측 - CNN 모델 *
#  
#
# hw_4_data.csv를 train, validation, test 로 구분한 데이터를 사용하여 리뷰글 긍부정(긍정:1, 부정:0) 예측하는 CNN 모델을 만드시기 바랍니다.
# 
# 모델을 만드실때 모델 구조, 하이퍼파라미터 등은 각자의 방법으로 정하시기 바랍니다.
#  
# 
# 모델 생성 후 보내 드린 데이터 파일 'hw_5_model_test_data_no_label.csv' 의 리뷰글에 대하여 긍부정으로 예측하여
#
# 이를 파일('hw_6_CNN_predict_label_본인이름.csv')로 저장하여 보내주시기 바랍니다.
#
# 
#
# 8/19 까지 1.작성한 코드, 2.'hw_6_CNN_predict_label_본인이름.csv' 파일 , 3.과제 발표할 PPT 를 메일로 보내주시기 바랍니다.
# 
# =============================================================================

import tensorflow as tf
from gensim.models.word2vec import Word2Vec
from konlpy.tag import Okt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from tqdm import tqdm
import os, pickle, time, sys


os.environ['TF_ENABLE_WINOGRAD_NONFUSED'] = '1'
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
#os.environ["CUDA_VISIBLE_DEVICES"] = '0'

class hw_CNN_model():
    def __init__(self, tf_model_path='', tf_model_name='', pred_hw_test_file_path='', pred_hw_test_file_name=''):
        self.tf_model_important_var_name = 'important_vars_ops'
        
        self.tf_model_path = tf_model_path
        self.tf_model_name = tf_model_name
        
        self.pred_hw_test_file_path = pred_hw_test_file_path
        self.pred_hw_test_file_name = pred_hw_test_file_name
        
        self.w2v_size = 100
        self.max_word_num = 50
                
        self.early_stopping_patience = 10
        self.epoch_num = 100
        self.batch_size = 512
        
        self.learning_rate = 0.001
        
    ###############################################################################
    #     data preprocessing
    ###############################################################################
    def load_data_fn(self, data_file_path, data_file_name, w2v_model):
        
        os.chdir(data_file_path)
        
        with open(data_file_name[1], 'rb') as f:
           raw_x, load_x, load_y = pickle.load(f)
        
        return raw_x, load_x, load_y
        
    ###############################################################################
    #     create model
    ###############################################################################
    def create_CNN_model(self):
        tf.reset_default_graph()
        
        x_data = tf.placeholder(tf.float32, [None, 50,100], name='x_data')
        re_x_data = tf.reshape(x_data, [-1, 50, 100, 1])
        y_data = tf.placeholder(tf.float32, [None, 2], name='y_data')
        
        filter_sizes = [3,4,5]
        # Create a convolution + maxpool layer for each filter size
        pooled_outputs = []
        for i, filter_size in enumerate(filter_sizes):
            with tf.name_scope("conv-maxpool-%s" % filter_size): 
                # Convolution Layer
                filter_shape = [filter_size,  self.w2v_size, 1, 64]
                W = tf.Variable(tf.random_normal(filter_shape, stddev=0.1), name="W")
                b = tf.Variable(tf.zeros(64), name="b")
                conv = tf.nn.conv2d(
                    re_x_data,
                    W,
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="conv")
                # Apply nonlinearity
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                # Maxpooling over the outputs
                pooled = tf.nn.max_pool(
                    h,
                    ksize=[1, self.max_word_num - filter_size + 1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="pool")
                pooled_outputs.append(pooled)
                
        # Combine all the pooled features
        for i, x in enumerate(pooled_outputs):
            pooled_outputs[i] = tf.reshape(x, [-1, 1*1*64])
        conv_output = tf.concat(pooled_outputs, axis=1)     
        num_filters_total = 64 * len(filter_sizes)
        
        fc_w_1 = tf.Variable(tf.random_normal([num_filters_total, 256], stddev=0.01))
        fc_b_1 = tf.Variable(tf.zeros(shape=[256]))
        fc_u_1 = tf.add(tf.matmul(conv_output, fc_w_1), fc_b_1)
        fc_z_1 = tf.nn.relu(fc_u_1)
        
        fc_w_2 = tf.Variable(tf.random_normal([256, 2], stddev=0.01))
        fc_u_2 = tf.matmul(fc_z_1, fc_w_2)
        
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=fc_u_2, labels=y_data), name='loss')
        optimizer = tf.train.AdamOptimizer(self.learning_rate)
        train = optimizer.minimize(loss)

        pred_y = tf.nn.softmax(fc_u_2, name='pred_y')
        pred = tf.equal(tf.argmax(pred_y, 1), tf.argmax(y_data, 1), name='pred')
        acc = tf.reduce_mean(tf.cast(pred, tf.float32), name='acc')
        
        for op in [x_data, y_data, loss, acc, pred_y]:
            tf.add_to_collection(self.tf_model_important_var_name, op)   
        
      
       
        return x_data, y_data, loss, acc, pred_y, train 
        
    ###############################################################################
    #     early_stopping_and_save_model
    ###############################################################################
    def early_stopping_and_save_model(self, sess, saver, input_vali_loss, early_stopping_val_loss_list):
        if len(early_stopping_val_loss_list) != self.early_stopping_patience:
            early_stopping_val_loss_list = [99.99 for _ in range(self.early_stopping_patience)]
        
        early_stopping_val_loss_list.append(input_vali_loss)
        if input_vali_loss < min(early_stopping_val_loss_list[:-1]):
            os.chdir(self.tf_model_path)
            saver.save(sess, './{0}/{0}.ckpt'.format(self.tf_model_name))
            early_stopping_val_loss_list.pop(0)
            
            return True, early_stopping_val_loss_list
        
        elif early_stopping_val_loss_list.pop(0) < min(early_stopping_val_loss_list):
            return False, early_stopping_val_loss_list
        
        else:
            return True, early_stopping_val_loss_list
        
    ###############################################################################
    #     model train
    ###############################################################################
    def model_train(self, x_train, y_train, x_vali, y_vali):
        x_data, y_data, loss, acc, pred_y, train = self.create_CNN_model()
        
        batch_index_list = list(range(0, x_train.shape[0], self.batch_size))
        vali_batch_index_list = list(range(0, x_vali.shape[0], self.batch_size))
        
        train_loss_list, vali_loss_list = [], []
        train_acc_list, vali_acc_list = [], []
        
        start_time = time.time()
        saver = tf.train.Saver()
        early_stopping_val_loss_list = []
        print('\n%s\n%s - training....'%('-'*100, self.tf_model_name))
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
                    
            for epoch in range(self.epoch_num):
                total_loss, total_acc, vali_total_loss, vali_total_acc = 0, 0, 0, 0
                processing_bar_var = [0, 0]
                
                train_random_seed = int(np.random.random()*10**4)
                for x in [x_train, y_train]:
                    np.random.seed(train_random_seed)
                    np.random.shuffle(x)
                    
                for i in batch_index_list:
                    batch_x, batch_y = x_train[i:i+self.batch_size], y_train[i:i+self.batch_size]
                    
                    processing_bar_var[0] += len(batch_x)
                    processing_bar_print = int(processing_bar_var[0]*100/len(x_train))-processing_bar_var[1]
                    if processing_bar_print != 0:
                        sys.stdout.write('-'*processing_bar_print)
                        sys.stdout.flush()
        
                    processing_bar_var[1] += (int(processing_bar_var[0]*100/len(x_train))-processing_bar_var[1])
                    
                    _, loss_val, acc_val = sess.run([train, loss, acc], feed_dict={x_data: batch_x, y_data: batch_y})
                    total_loss += loss_val
                    total_acc += acc_val
                    
                train_loss_list.append(total_loss/len(batch_index_list))
                train_acc_list.append(total_acc/len(batch_index_list))
                
                sys.stdout.write('\n#%4d/%d%s' % (epoch + 1, self.epoch_num, '  |  '))
                sys.stdout.write('Train_loss={:.4f} / Train_acc={:.4f}{}'.format(train_loss_list[-1], train_acc_list[-1], '  |  '))
                sys.stdout.flush()
                        
                for i in vali_batch_index_list:
                    vali_batch_x, vali_batch_y = x_vali[i:i+self.batch_size], y_vali[i:i+self.batch_size]
                    
                    vali_loss_val, vali_acc_val = sess.run([loss,acc], feed_dict={x_data: vali_batch_x, y_data: vali_batch_y})
                    vali_total_loss += vali_loss_val
                    vali_total_acc += vali_acc_val
                        
                vali_loss_list.append(vali_total_loss/len(vali_batch_index_list))
                vali_acc_list.append(vali_total_acc/len(vali_batch_index_list))
                
                tmp_running_time = time.time() - start_time
                sys.stdout.write('Vali_loss={:.4f} / Vali_acc={:.4f}{}'.format(vali_loss_list[-1], vali_acc_list[-1], '  |  '))
                sys.stdout.write('%dm %5.2fs\n'%(tmp_running_time//60, tmp_running_time%60))
                sys.stdout.flush()
                
                bool_continue, early_stopping_val_loss_list = self.early_stopping_and_save_model(sess, saver, vali_loss_list[-1], early_stopping_val_loss_list)
                if not bool_continue:
                    print('{0}\nstop epoch : {1}\n{0}'.format('-'*100, epoch-self.early_stopping_patience+1))
                    break
                
        
        running_time = time.time() - start_time
        print('%s\ntraining time : %d m  %5.2f s\n%s'%('*'*100, running_time//60, running_time%60, '*'*100))
        
        
        os.chdir(r'{}\{}'.format(self.tf_model_path, self.tf_model_name))
        epoch_list = [i for i in range(1, epoch+2)]
        graph_loss_list = [train_loss_list, vali_loss_list, 'r', 'b', 'loss', 'upper right', '{}_loss.png'.format(self.tf_model_name)]
        graph_acc_list = [train_acc_list, vali_acc_list, 'r--', 'b--', 'acc', 'lower right', '{}_acc.png'.format(self.tf_model_name)]
        for train_l_a_list, vali_l_a_list, trian_color, vali_color, loss_acc, legend_loc, save_png_name in [graph_loss_list, graph_acc_list]:
            plt.plot(epoch_list, train_l_a_list, trian_color, label='train_'+loss_acc)
            plt.plot(epoch_list, vali_l_a_list, vali_color, label='validation_'+loss_acc)
            plt.xlabel('epoch')
            plt.ylabel(loss_acc)
            plt.legend(loc=legend_loc)
            plt.title(self.tf_model_name)
            plt.savefig(save_png_name)
            plt.show()
        
    ###############################################################################
    #     model test
    ###############################################################################
    def model_test(self, x_test, y_test):
        tf.reset_default_graph()
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            
            os.chdir(self.tf_model_path)
            saver = tf.train.import_meta_graph('./{0}/{0}.ckpt.meta'.format(self.tf_model_name))
            saver.restore(sess, './{0}/{0}.ckpt'.format(self.tf_model_name))
            x_data, y_data, loss, acc, pred_y = tf.get_collection(self.tf_model_important_var_name)
            
            test_loss, test_acc, test_pred_y, test_true_y = sess.run([loss, acc, pred_y, y_data], feed_dict={x_data: x_test, y_data: y_test})
            
        f_pred_y = np.argmax(test_pred_y, axis=1)
        f_true_y = np.argmax(test_true_y, axis=1)
        print('\n' + '='*100)
        print(classification_report(f_true_y, f_pred_y, target_names=['neg/pos_'+str(i) for i in range(2)]))
        print(pd.crosstab(pd.Series(f_true_y), pd.Series(f_pred_y), rownames=['True'], colnames=['Predicted'], margins=True))
        print('\n{}\nTest_loss = {:.4f}\nTest_acc = {:.4f}\n{}'.format('='*100, test_loss, test_acc, '='*100))
            
    ###############################################################################
    #     predict label
    ###############################################################################
    def predict_label(self, raw_hw_test, x_hw_test):
        tf.reset_default_graph()
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            
            os.chdir(self.tf_model_path)
            saver = tf.train.import_meta_graph('./{0}/{0}.ckpt.meta'.format(self.tf_model_name))
            saver.restore(sess, './{0}/{0}.ckpt'.format(self.tf_model_name))
            x_data, y_data, loss, acc, pred_y = tf.get_collection(self.tf_model_important_var_name)
            
            hw_y_pred = sess.run(pred_y, feed_dict={x_data: x_hw_test})
            f_pred_y = np.argmax(hw_y_pred, axis=1)
            
        os.chdir(self.pred_hw_test_file_path)
        pd.DataFrame({'label':f_pred_y, 'review':raw_hw_test}).to_csv(self.pred_hw_test_file_name, index=False, encoding='cp949')
            

if __name__ == "__main__": 
# =============================================================================
    data_file_path = r'C:\Users\CHOYEONGKYU\Desktop\프학 과제\4주차'
    train_file_name = ['hw_4_train_data_CYK.csv', 'hw_5_train_data_pickle_CYK.pkl']
    validation_file_name = ['hw_4_validation_data_CYK.csv', 'hw_5_vali_data_pickle_CYK.pkl']
    test_file_name = ['hw_4_test_data_CYK.csv','hw_5_test_data_pickle_CYK.pkl']
    
    hw_test_file_path = r'C:\Users\CHOYEONGKYU\Desktop\프학\5주차'
    hw_test_file_name = ['hw_5_model_test_data_no_label.csv', 'hw_5_model_test_data_no_label_pickle_CYK.pkl']
    w2v_file_path = r'C:\Users\CHOYEONGKYU\Desktop\프학 과제\4주차'
    os.chdir(w2v_file_path)
    w2v_model = Word2Vec.load('hw_4_word2vec_CYK.model')
# =============================================================================    
    tf_model_path = r'C:\Users\CHOYEONGKYU\Desktop\프학 과제\6주차'
    tf_model_name = 'hw_6_CNN_model_CYK'
    
    pred_hw_test_file_path = r'C:\Users\CHOYEONGKYU\Desktop\프학 과제\6주차'
    pred_hw_test_file_name = 'hw_6_CNN_predict_label_CYK.csv'
# =============================================================================   
    
    hw_CNN = hw_CNN_model(tf_model_path, tf_model_name, pred_hw_test_file_path, pred_hw_test_file_name)

    _, x_train, y_train = hw_CNN.load_data_fn(data_file_path, train_file_name, w2v_model)
    _, x_vali, y_vali = hw_CNN.load_data_fn(data_file_path, validation_file_name, w2v_model)
    _, x_test, y_test = hw_CNN.load_data_fn(data_file_path, test_file_name, w2v_model)
    raw_hw_test, x_hw_test, _ = hw_CNN.load_data_fn(hw_test_file_path, hw_test_file_name, w2v_model)
    
    print('train_shape : {} / {}'.format(x_train.shape, y_train.shape))
    print('validation_shape : {} / {}'.format(x_vali.shape, y_vali.shape))
    print('test_shape : {} / {}'.format(x_test.shape, y_test.shape))
    print('\nhw_test_shape : {}'.format(x_hw_test.shape))
    
    hw_CNN.model_train(x_train, y_train, x_vali, y_vali)
    hw_CNN.model_test(x_test, y_test)
    hw_CNN.predict_label(raw_hw_test, x_hw_test)
    