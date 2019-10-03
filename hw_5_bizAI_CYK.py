# -*- coding: utf-8 -*-

# =============================================================================
# # BIZ&AI lab 6기 과제_5
# 
# 
# * 리뷰글 긍부정 예측 - ANN 모델 *
#  
#
# hw_4_data.csv를 train, validation, test 로 구분한 데이터를 사용하여 리뷰글 긍부정(긍정:1, 부정:0) 예측하는 ANN 모델을 만드시기 바랍니다.
# 
# 모델을 만드실때 모델 구조, 하이퍼파라미터 등은 각자의 방법으로 정하시기 바랍니다.
#  
# 
# 모델 생성 후 보내 드린 데이터 파일 'hw_5_model_test_data_no_label.csv' 의 리뷰글(input)에 대하여 긍부정(output)으로 예측하여
#
# 이를 파일('hw_5_ann_predict_label_본인이름.csv')로 저장하여 보내주시기 바랍니다.
#
# 
#
# 8/8 까지 1.작성한 코드, 2.'hw_5_ann_predict_label_본인이름.csv' 파일 , 3.과제 발표할 PPT 를 메일로 보내주시기 바랍니다.
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

class hw_ANN_model():
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
        
        self.num_class = 2
        self.dropout_rate = 0.5
        self.learning_rate = 0.001
        
        self.layer_1_node = 512
        self.layer_2_node = 256

###############################################################################
#     data preprocessing
###############################################################################
    def load_data_fn(self, data_file_path, data_file_name, w2v_model):
        os.chdir(data_file_path)
        raw_data = pd.read_csv(data_file_name[0], encoding='cp949')    
        
        wv = w2v_model.wv    
        wv_result=[]
        
        okt = Okt()
        result=[]
        pos = ['Verb', 'Adjective', 'Noun','Josa', 'Eomi','Conjunction','Adverb','Determiner']
        for x in range(len(raw_data)):
            words = okt.pos(raw_data.loc[x,'review'])
            tmp_result=[]
            for y,z in words:
                if z in pos: tmp_result.append(y)
            result.append(tmp_result)     
                
        for i,data in tqdm(enumerate(result)): #result는 형태소 분석된 총집합. 리스트 형태
            if not (i+1)%1000:
                print(data_file_name[0]+'\t', i+1, ' / ', raw_data.shape[0])
                
            tmp_wv = [] #  1개의 data의 단어들의 벡터값 모음. 50*100
            if len(data) > self.max_word_num: 
                data = data[:self.max_word_num] #50단어 까지만 받기.         
            for x in data: 
                try:
                    tmp_wv.append(wv[str(x)])    # 단어에 대한 벡터 구하기. ndarray
                except:
                    tmp_wv.append(np.zeros(self.w2v_size))
                    
            if len(tmp_wv)<self.max_word_num: 
                zeros=[]
                zeros += [np.zeros(self.w2v_size) for _ in range(self.max_word_num-len(tmp_wv))]
                wv_result.append(zeros+tmp_wv)
            else: wv_result.append(tmp_wv)
        
        try:
            labels = raw_data.loc[:,'label']  
            label_hot = tf.constant(labels, name='label_hot')     
            sess = tf.Session()
            sess.run(tf.global_variables_initializer())
            label_one_hot = sess.run(tf.one_hot(label_hot, depth=2))
            load_y = np.asarray(label_one_hot)
            sess.close()
        except:
            load_y = None
        
        raw_x = raw_data.loc[:,'review']
        load_x = np.asarray(wv_result)
        
        with open(data_file_name[1], 'wb') as f:
            pickle.dump([raw_x, load_x, load_y], f)       
        
        return raw_x, load_x, load_y
    
###############################################################################
#     create model
###############################################################################
    def create_ANN_model(self):
        tf.reset_default_graph()
        
        x_data = tf.placeholder(tf.float32, [None, self.max_word_num, self.w2v_size], name='x_data')
        re_x_data = tf.reshape(x_data, [-1, self.max_word_num*self.w2v_size])
        y_data = tf.placeholder(tf.float32, [None, self.num_class], name='y_data')
        
        he_init = tf.contrib.layers.variance_scaling_initializer() # relu를 쓸 때는 'he'라는 방식으로 초기 weight값을 설정해주면 좋다.
        
        z_1 = tf.layers.dense(re_x_data, self.layer_1_node, activation=tf.nn.relu, kernel_initializer=he_init)
        d_1 = tf.nn.dropout(z_1, self.dropout_rate, name='d_1')
        
        w_2 = tf.Variable(tf.random_normal([self.layer_1_node, self.layer_2_node], stddev=0.01), name='w_2')
        b_2 = tf.Variable(tf.zeros(shape=(self.layer_2_node)), name='b_2')
        bn_2 = tf.layers.batch_normalization(tf.matmul(d_1, w_2)+b_2, name='bn_2')
        z_2 = tf.nn.relu(bn_2, name='z_2')
        d_2 = tf.nn.dropout(z_2, self.dropout_rate, name='d_2')
        
        w_3 = tf.Variable(tf.random_normal([self.layer_2_node, self.num_class], stddev=0.01), name='w_3')
        u_3 = tf.matmul(d_2, w_3, name='u_3')
    
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=u_3, labels=y_data), name='loss')
        optimizer = tf.train.AdamOptimizer(self.learning_rate)
        train = optimizer.minimize(loss)

        pred_y = tf.nn.softmax(u_3, name='pred_y')
        pred = tf.equal(tf.argmax(pred_y, 1), tf.argmax(y_data, 1), name='pred')
        acc = tf.reduce_mean(tf.cast(pred, tf.float32), name='acc')
        
        for op in [x_data, y_data, loss, acc, pred_y]:
            tf.add_to_collection(self.tf_model_important_var_name, op)
    
        return x_data, y_data, loss, acc, pred_y, train
    
###############################################################################
#     model train
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
        
        elif early_stopping_val_loss_list.pop(0) < min(early_stopping_val_loss_list): # a 맨 앞꺼 빠짐. 가장 앞에 있는게 가장 작으면 10번 올라간거임. 
            return False, early_stopping_val_loss_list
        
        else:
            return True, early_stopping_val_loss_list
    

    def model_train(self, x_train, y_train, x_vali, y_vali):
        x_data, y_data, loss, acc, pred_y, train = self.create_ANN_model() # creat ANN 모델 실행. 
        
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
            
        f_true_y = np.argmax(test_true_y, axis=1)
        f_pred_y = np.argmax(test_pred_y, axis=1)
        print('\n' + '='*100)
        print(classification_report(f_true_y, f_pred_y, target_names=['Positive', 'Negative']))
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
            
    ###############################################################################
    #     fn_hw_ai
    ###############################################################################
    def fn_hw_ai(self, true_hw_test_file_path, true_hw_test_file_name, hw_ai_file_path, hw_ai_file_name_list):
        os.chdir(true_hw_test_file_path)
        model_y_true = pd.read_csv(true_hw_test_file_name, encoding='cp949')['label'].values
        
        for tmp_pred_label_file_name in [self.pred_hw_test_file_name] + hw_ai_file_name_list:
            if tmp_pred_label_file_name == self.pred_hw_test_file_name:
                os.chdir(self.pred_hw_test_file_path)
            else:
                os.chdir(hw_ai_file_path)
                
            print('%s\n%s\n'%('#'*100, tmp_pred_label_file_name))
            
            try:
                model_y_pred = pd.read_csv(tmp_pred_label_file_name, engine='python')['label'].values
                
                print(classification_report(model_y_true, model_y_pred, target_names=['Positive', 'Negative']))
                print(pd.crosstab(pd.Series(model_y_true), pd.Series(model_y_pred), rownames=['True'], colnames=['Predicted'], margins=True))
                
                model_test_acc = np.sum(np.equal(model_y_true, model_y_pred)) / np.shape(model_y_true)[0]
                print('\nmodel_test_acc = {:.4f}'.format(model_test_acc))
            except FileNotFoundError as e:
                print(e)
            except ValueError as e:
                print(e)
        print('#'*100)
        

        
        
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
    tf_model_path = r'C:\Users\CHOYEONGKYU\Desktop\프학 과제\5주차'
    tf_model_name = 'hw_5_ANN_model_CYK'
    
    tf_model_important_var_name = 'important_vars_ops'
    
    pred_hw_test_file_path = r'C:\Users\CHOYEONGKYU\Desktop\프학 과제\5주차'
    pred_hw_test_file_name = 'hw_5_ann_predict_label_CYK.csv'
# =============================================================================  
    
# =============================================================================   
   
    hw_ANN = hw_ANN_model(tf_model_path, tf_model_name, pred_hw_test_file_path, pred_hw_test_file_name) # 인스턴스를 만듦. 

    try:
        os.chdir(data_file_path)
        with open(train_file_name[1], 'rb') as f:
            _, x_train, y_train = pickle.load(f)
        with open(validation_file_name[1], 'rb') as f:
            _, x_vali, y_vali = pickle.load(f)
        with open(test_file_name[1], 'rb') as f:
            _, x_test, y_test = pickle.load(f)
    
        os.chdir(hw_test_file_path)
        with open(hw_test_file_name[1], 'rb') as f:
            raw_hw_test, x_hw_test, _ = pickle.load(f)
    except FileNotFoundError:
        _, x_train, y_train = hw_ANN.load_data_fn(data_file_path, train_file_name, w2v_model)
        _, x_vali, y_vali = hw_ANN.load_data_fn(data_file_path, validation_file_name, w2v_model)
        _, x_test, y_test = hw_ANN.load_data_fn(data_file_path, test_file_name, w2v_model)
        raw_hw_test, x_hw_test, _ = hw_ANN.load_data_fn(hw_test_file_path, hw_test_file_name, w2v_model)
        
    print('{}\ntrain_shape : {} / {}'.format('#'*100, x_train.shape, y_train.shape))
    print('validation_shape : {} / {}'.format(x_vali.shape, y_vali.shape))
    print('test_shape : {} / {}'.format(x_test.shape, y_test.shape))
    print('\nmodel_test_shape : {}\n{}'.format(x_hw_test.shape, '#'*100))
    
    hw_ANN.model_train(x_train, y_train, x_vali, y_vali)
    hw_ANN.model_test(x_test, y_test)
    hw_ANN.predict_label(raw_hw_test, x_hw_test)
    
# =============================================================================
    hw_ANN.fn_hw_ai(true_hw_test_file_path, true_hw_test_file_name, hw_ai_file_path, hw_ai_file_name_list)
# =============================================================================
    