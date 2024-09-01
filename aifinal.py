import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import Counter
import tensorflow as tf
from tensorflow import keras
from keras import layers
                                                   
# Path to 指定路徑
data_dir = Path("D:/AI Final/samples")

#獲取所有圖片列表

#使用 glob 搜尋指定路徑 data_dir 下所有的 .png 檔案。由list() 返回符合的
#map(str, ...) 將檔案路徑轉換成字串型態， sorted 對這些檔案進行排序
images = sorted(list(map(str, list(data_dir.glob("*.png")))))
#使用 split(os.path.sep) 將檔案路徑根據路徑分隔符號（在這裡使用 os.path.sep 來獲取正確的分隔符號）進行拆分，然後選取最後一個元素，即該檔案的檔名部分
#接著，使用 split(".png") 以 .png 作為分隔符號拆分，取得最後的部分，即該檔案的標籤。最終，將這些標籤組成一個列表
labels = [img.split(os.path.sep)[-1].split(".png")[0] for img in images]
#使用集合推導式，在每個標籤中遍歷字符，並將這些字符收集到一個集合中。集合會自動去除重複的字符，因此結果集合中只包含標籤中的唯一字符
characters = set(char for label in labels for char in label)
#將集合轉換為列表，並使用 sorted() 對列表進行排序
#排序後 characters 包含標籤中唯一的字符，並以排序後的形式呈現
characters = sorted(list(characters)) 
print("Number of images found: ", len(images))#圖片數量
print("Number of labels found: ", len(labels))#標籤數量
print("Number of unique characters: ", len(characters))#唯一字符的數量
print("Characters present: ", characters) #列出所有的字符


batch_size = 16 #訓練和測試的批量大小
#所需的圖像尺寸
img_width = 200
img_height = 50

#使用兩個卷積層，每個塊都有一個池化層，將特徵下採樣 2 倍
# 因此總採樣因子為 4
downsample_factor = 4

max_length = max([len(label) for label in labels]) #計算標籤中最長的字元數量

#將字符映射為數字的 StringLookup 層(整數), None，表示不使用遮罩標記
char_to_num = layers.StringLookup(
    vocabulary=list(characters), mask_token=None
)

#將整數映射回原始字符
num_to_char = layers.StringLookup(
    vocabulary=char_to_num.get_vocabulary(), mask_token=None, invert=True
)

#訓練集設定為0.9;shuffle：是對數據進行洗牌
def split_data(images, labels, train_size=0.9, shuffle=True):
    size = len(images)# 1. 獲取數據集的總大小
    # 2. 創建一個索引數組並對其洗牌
    indices = np.arange(size)
    if shuffle:
        np.random.shuffle(indices)
    # 3. 獲取訓練樣本的大小
    train_samples = int(size * train_size)
   # 4. 將數據拆分為訓練集和測試集
    x_train, y_train = images[indices[:train_samples]], labels[indices[:train_samples]]
    x_valid, y_valid = images[indices[train_samples:]], labels[indices[train_samples:]]
    return x_train, x_valid, y_train, y_valid

# 將數據拆分為訓練集和測試集
x_train, x_valid, y_train, y_valid = split_data(np.array(images), np.array(labels))


def encode_single_sample(img_path, label):
    
    img = tf.io.read_file(img_path)# 1. 讀取圖片
    img = tf.io.decode_png(img, channels=1) #解碼並轉換為灰度    
    img = tf.image.convert_image_dtype(img, tf.float32)#在[0, 1]範圍內轉換為float32
    img = tf.image.resize(img, [img_height, img_width])#調整到想要的大小200*50
    # 轉置圖像，將需要尺寸對應於圖像的寬度
    #perm=[1, 0, 2]：轉置後的維度順序。根據指定的順序，第一個維度將變為第二個維度，第二個維度將變為第一個維度，而第三個維度保持不變
    img = tf.transpose(img, perm=[1, 0, 2])
    #將標籤中的字符映射到數字
    label = char_to_num(tf.strings.unicode_split(label, input_encoding="UTF-8"))
   #返回一個字典，模型需要兩個輸入
    return {"image": img, "label": label}

#建立訓練用的 TensorFlow Dataset
train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_dataset = (
    #使用 map() 函式對數據集中的每個樣本應用 encode_single_sample 函式
    #這個函式可能是自定義的編碼處理邏輯，用於對圖片和標籤進行處理和轉換
    #num_parallel_calls=tf.data.AUTOTUNE 表示自動選擇最佳值
    train_dataset.map(
        encode_single_sample, num_parallel_calls=tf.data.AUTOTUNE
    )
    .batch(batch_size)#將數據集中的樣本按照 batch_size 的大小進行分批處理，將一批一批的樣本作為模型的輸入
    #prefetch()增加數據的預取功能，從而在模型訓練時可以同步讀取數據和進行計算，提高訓練效率
    #buffer_size=tf.data.AUTOTUNE 表示自動選擇最佳的緩衝區大小
    .prefetch(buffer_size=tf.data.AUTOTUNE)
)

#建立測試用的 TensorFlow Dataset
validation_dataset = tf.data.Dataset.from_tensor_slices((x_valid, y_valid))
validation_dataset = (
    validation_dataset.map(
        encode_single_sample, num_parallel_calls=tf.data.AUTOTUNE
    )
    .batch(batch_size)
    .prefetch(buffer_size=tf.data.AUTOTUNE)
)

_, ax = plt.subplots(4, 4, figsize=(10, 10))#創建一個 4x4 的子圖格子，用於顯示圖片和標籤
#使用 take(1) 方法從訓練數據集中取出一個批次的數據
for batch in train_dataset.take(1):
    images = batch["image"] #獲取批次中的圖片張量
    labels = batch["label"] #獲取批次中的標籤張量
    for i in range(16):
        img = (images[i] * 255).numpy().astype("uint8")#將圖片張量轉換為 numpy 數組形式，並將像素值乘以 255，然後轉換為"uint8"
        label = tf.strings.reduce_join(num_to_char(labels[i])).numpy().decode("utf-8")#將標籤張量轉換為字符串形式，並使用 num_to_char 函式將數字標籤轉換為字符標籤
        #在子圖格子中顯示圖片，img[:, :, 0] 表示提取圖片的灰度通道，
        # .T 表示轉置圖片以符合預期的顯示方式，cmap="gray" 指定使用灰度色彩映射
        ax[i // 4, i % 4].imshow(img[:, :, 0].T, cmap="gray")
        ax[i // 4, i % 4].set_title(label)#設定子圖格子的標題為對應的標籤
        ax[i // 4, i % 4].axis("off")#隱藏子圖格子的座標軸
plt.show() #顯示圖片和標籤

#使用CTCLayer 類別繼承自 layers.Layer，並在 __init__ 方法中初始化了 loss_fn 屬性
class CTCLayer(layers.Layer):
    def __init__(self, name=None):
        super().__init__(name=name)
        self.loss_fn = keras.backend.ctc_batch_cost
    #用於計算訓練時間的損失值並將其添加到層中
    def call(self, y_true, y_pred):
        # 計算訓練時間損失值並使用 `self.add_loss()` 將其添加到層中。
        batch_len = tf.cast(tf.shape(y_true)[0], dtype="int64")#獲取批次數據的大小。
        input_length = tf.cast(tf.shape(y_pred)[1], dtype="int64")#獲取測試序列的長度。
        label_length = tf.cast(tf.shape(y_true)[1], dtype="int64")#獲取真實標籤序列的長度。
        input_length = input_length * tf.ones(shape=(batch_len, 1), dtype="int64")#將測試序列的長度擴展為與批次數據大小相同的樣子
        label_length = label_length * tf.ones(shape=(batch_len, 1), dtype="int64")#將真實標籤序列的長度擴展為與批次數據大小相同的
        loss = self.loss_fn(y_true, y_pred, input_length, label_length) #計算 CTC 損失
        self.add_loss(loss)#將計算得到的損失值添加到層中，以便在模型訓練時進行梯度下降
        return y_pred #在測試時，直接返回計算得到的預測值

#建立 model
def build_model():

    input_img = layers.Input(
        shape=(img_width, img_height, 1), name="image", dtype="float32" # '1'假設圖片是灰度圖
    )
    labels = layers.Input(name="label", shape=(None,), dtype="float32")#標籤的輸入使用了 shape=(None,)表示標籤的長度是可變的。

    # 第一層捲積層
    x = layers.Conv2D(  #使用二維捲基層
        32, #輸出的通道數
        (3, 3),  #卷積核的大小，這裡是 3x3
        activation="relu",
        kernel_initializer="he_normal", #指定卷積核的初始化方法為he_normal
        padding="same", #輸入和輸出的大小相同，使用零填充
        name="Conv1", #卷積層的名
    )(input_img) #將輸入層 input_img 作為輸入數據
    x = layers.MaxPooling2D((2, 2), name="pool1")(x) #二維最大池化層，指定池化的大小 2x2，(x)：將前一個卷積層的輸出 x 作為輸入數據

    # 第二層捲積層
    x = layers.Conv2D(
        64,
        (3, 3),
        activation="relu",
        kernel_initializer="he_normal",
        padding="same",
        name="Conv2",
    )(x)#將前一個池化層的輸出 x 作為輸入數據。
    x = layers.MaxPooling2D((2, 2), name="pool2")(x)


    # 使用了兩個池大小和步幅為 2 的最大池。
    # 因此，下採樣的特徵圖小4 倍。最後一層的過濾器數量為64。在將輸出傳遞給模型的RNN部分之前進行相應的整形
    new_shape = ((img_width // 4), (img_height // 4) * 64)
    x = layers.Reshape(target_shape=new_shape, name="reshape")(x)#重塑層，用於改變張量的形狀。(x)：將前一個池化層的輸出 x 作為輸入數據。
    x = layers.Dense(64, activation="relu", name="dense1")(x)#'64'：指定輸出的維度大小。(x)：將前一個重塑層的輸出 x 作為輸入數據。
    x = layers.Dropout(0.2)(x)#指定丟棄的比例20%。(x)：將前一個全連接層的輸出 x 作為輸入數據。

    # RNNs
    
    #layers.Bidirectional：這是一個雙向層，用於將輸入數據同時傳遞給 LSTM 層的正向和反向部分
    #'128'：指定 LSTM 層的隱藏狀態的維度大小
    #return_sequences=True：指定返回每個時間步的輸出序列，而不是只返回最後一個時間步的輸出。dropout=0.25：指定丟棄的比例25%
    #(x)：將前一個層的輸出 x 作為輸入數據  
    x = layers.Bidirectional(layers.LSTM(128, return_sequences=True, dropout=0.25))(x)
    x = layers.Bidirectional(layers.LSTM(64, return_sequences=True, dropout=0.25))(x)


   
    # 輸出層
    x = layers.Dense(
        #輸出的維度大小，這裡是字符數量加一，加一是為了包含 CTC loss 的 blank 符號
        len(char_to_num.get_vocabulary()) + 1, activation="softmax", name="dense2"
    )(x)#(x)：將前一個層的輸出 x 作為輸入數據

    
    # 添加CTC層用於計算每一步的CTC損失
    output = CTCLayer(name="ctc_loss")(labels, x)

    
    # 定義模型
    model = keras.models.Model(
        inputs=[input_img, labels], outputs=output, name="ocr_model_v1"
    )
    
# 優化器
    opt = keras.optimizers.Adam()#Adam 是常用的優化器，用於調整模型的參數以最小化損失函數
    # 編譯模型並返回
    model.compile(optimizer=opt)
    return model


# 進入model
model = build_model() #創建模型
model.summary() #打印模型的摘要信息
epochs = 100 #訓練的輪數
early_stopping_patience = 10 #設置提前停止的耐心值，表示如果在連續的 10 輪訓練中驗證集的損失函數沒有改善，則提前停止訓練
# 添加提前停止
early_stopping = keras.callbacks.EarlyStopping(
    monitor="val_loss", patience=early_stopping_patience, restore_best_weights=True
)

#訓練模型
history = model.fit(
    train_dataset,  #訓練數據集，用於模型的訓練
    validation_data=validation_dataset,  #驗證數據集，用於在每個訓練輪結束後評估模型的性能
    epochs=epochs,
    callbacks=[early_stopping],  #回調函數的列表，包括提前停止的回調函數
)

# 通過提取層直到輸出層得到測試模型

prediction_model = keras.models.Model(
    model.get_layer(name="image").input, model.get_layer(name="dense2").output
)
prediction_model.summary() #打印出預測模型的摘要信息
#解碼模型的預測結果
def decode_batch_predictions(pred):
    input_len = np.ones(pred.shape[0]) * pred.shape[1]    #pred.shape[1]這是在 CTC 解碼過程中，需要指定輸入序列的長度。
    # 使用貪心搜索
    results = keras.backend.ctc_decode(pred, input_length=input_len, greedy=True)[0][0][
        :, :max_length  
        #results 是 CTC 解碼的結果，是一個形狀為 (batch_size, max_length) 的張量，其中每個元素代表著預測的字符索引
        #通過切片操作 [:, :max_length] 來截取序列的前 max_length 個字符
    ]
    
    # 遍歷結果並取回文本
    output_text = []  #存儲每個解碼結果的字符文本
    for res in results:
        res = tf.strings.reduce_join(num_to_char(res)).numpy().decode("utf-8") #tf.strings.reduce_join 是 TensorFlow 的函數，用於將字符索引序列轉換為字符文本
        #這裡的 num_to_char 是之前建立的將字符索引映射為字符的查詢表
        output_text.append(res)
    return output_text

# 檢查一些測試樣本的結果
for batch in validation_dataset.take(1):
    batch_images = batch["image"]
    batch_labels = batch["label"]

    preds = prediction_model.predict(batch_images)
    pred_texts = decode_batch_predictions(preds)

    orig_texts = []
    for label in batch_labels:
        label = tf.strings.reduce_join(num_to_char(label)).numpy().decode("utf-8")
        orig_texts.append(label)

    _, ax = plt.subplots(4, 4, figsize=(15, 5))
    for i in range(len(pred_texts)):
        img = (batch_images[i, :, :, 0] * 255).numpy().astype(np.uint8)
        img = img.T
        title = f"Prediction: {pred_texts[i]}"
        ax[i // 4, i % 4].imshow(img, cmap="gray")
        ax[i // 4, i % 4].set_title(title)
        ax[i // 4, i % 4].axis("off")
plt.show()