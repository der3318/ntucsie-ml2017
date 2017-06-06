## 機器學習作業六 報告
| 學號 | 系級 | 姓名 |
|:-:|:-:|:-:|
| B03902015 | 資工三 | 簡瑋德 |
#### 1. 請比較有無「normalize(rating)」的差別。並說明如何「normalize」。
* 模型的架構
    | Layer | Parameters | Name | Input |
    |:-:|:-:|:-:|:-:|
    | Input | `shape=(1,)` | userId | - |
    | Input | `shape=(1,)` | movieId | - |
    | Embedding | `input_dim=6041`, `output_dim=8`, `input_length=1` | uEmb | userId |
    | Flatten | - | uVec | uEmb |
    | Embedding | `input_dim=3953`, `output_dim=8`, `input_length=1` | mEmb | movieId |
    | Flatten | - | mVec | mEmb |
    | Embedding | `input_dim=6041`, `output_dim=1`, `input_length=1` | uEmb2 | userId |
    | Flatten | - | uBias | uEmb2 |
    | Embedding | `input_dim=3953`, `output_dim=8`, `input_length=1` | mEmb2 | movieId |
    | Flatten | - | mBias | mEmb2 |
    | dot | `[uVec, mVec]`, `axes=-1` | umDot | - |
    | add | `[umDot, uBias, mBias]` | output | - |
* 「Normalize」的方式 - 把「rate」除以$5$(最大值)
* 實驗參數
    * Batch Size = $128$
    * Epochs = $20$
    * Validation Split = $10\%$
* 實驗結果
    | 有無正規化 | 訓練過程 | 最低MSE |
    |:-:|:-:|:-:|
    | 無 | ![Imgur](http://i.imgur.com/iEtGCxF.png) | 0.7564 |
    | 有 | ![Imgur](http://i.imgur.com/fIYWCV4.png) | 0.7737 |
* 觀察和比較
    * 有做正規化的模型，不論在「Train」或「Validation」損失都下降得很快，約在四個「epochs」時「Validation」就收斂
    * 沒做正規化的模型，雖然下降得較慢，但反而有比較好的結果
    * 兩個模型的參數、架構相同，只差在「Loss」的計算，進而影響「Gradient」的大小，結果卻相差甚大


#### 2. 比較不同的「latent dimension」的結果。
* 模型架構同上一題
* 實驗參數
    * Batch Size = $128$
    * Epochs = $12$
    * Validation Split = $10\%$
* 實驗結果
![Imgur](http://i.imgur.com/gXH0lla.png =500x400)
* 觀察和比較
    * 「Latent Dimension」是$8$或$16$的時候結果較好
    * 維度太低的話，損失不容易降低；維度超過$16$，則很容易「overfit」，可能需要搭配「early-stopping」或「regularization」來維護「Validation」和「Test」的表現


#### 3. 比較有無「bias」的結果。
* 模型架構同第一題
* 實驗參數
    * Batch Size = $128$
    * Epochs = $20$
    * Validation Split = $10\%$
* 實驗結果
    | 有無偏置 | 訓練過程 | 最低MSE |
    |:-:|:-:|:-:|
    | 有 | ![Imgur](http://i.imgur.com/iEtGCxF.png) | 0.7564 |
    | 無 | ![Imgur](http://i.imgur.com/IbKGPvQ.png) | 0.7589 |
* 觀察和比較
    * 有偏置的模型參數稍微多了一些，「Train」的損失降得比較低
    * 在「Validation」上的表現，兩者相差不大，「MSE」都能降到$0.75$左右


#### 4. 請試著用「DNN」來解決這個問題，並且說明實做的方法(方法不限)。並比較「MF」和「NN」的結果，討論結果的差異。
* 模型架構
    | Layer | Parameters | Name | Input |
    |:-:|:-:|:-:|:-:|
    | Input | `shape=(1,)` | userId | - |
    | Input | `shape=(1,)` | movieId | - |
    | Embedding | `input_dim=6041`, `output_dim=8`, `input_length=1` | uEmb | userId |
    | Flatten | - | uVec | uEmb |
    | Embedding | `input_dim=3953`, `output_dim=8`, `input_length=1` | mEmb | movieId |
    | Flatten | - | mVec | mEmb |
    | concatenate | `[uVec, mVec]` | concat | - |
    | Dense | `units=1` | output | concat |
* 實驗參數
    * Batch Size = $128$
    * Epochs = $20$
    * Validation Split = $10\%$
* 實驗結果
    | 架構 | 訓練過程 | 最低MSE |
    |:-:|:-:|:-:|
    | FM | ![Imgur](http://i.imgur.com/iEtGCxF.png) | 0.7564 |
    | DNN | ![Imgur](http://i.imgur.com/DH3Hr5W.png) | 0.8233 |
* 觀察和比較
    * 同樣的參數量，「DNN」明顯「underfit」，不論在「Train」或是「Validation」，「MSE」都超過$0.8$
    * 從實驗結果，可以得知，比起「concat」，「dot」本身有「consine-similarity」的意義，足以代表使用者和電影的相容性


#### 5. 請試著將「movie」的「embedding」用「tsne」降維後，將「movie category」當作「label」來作圖
* 我選了三個比較特別的類別：「Thriller」、「Chrilden's」和「Documentary」
* 作圖結果
![Imgur](http://i.imgur.com/AOBFPpn.png =650x500)
* 觀察和比較
    * 「Documentary」特別集中，我想它應該是最特別的一類電影，「Embedding」自然特別突出
    * 「Thriller」幾乎到處都有散布，可能是因為，不論甚麼樣的電影，多少都能夠加入一些恐怖、嚇人的元素，只是程度的差別，所以沒有很明顯的被區分出來
    * 「Chrildren's」則落在偏左上角的部分。因為兒童電影的種類可能還能再細分，所以比起「Documentary」，它的分布比較沒那麼集中


#### 6. 試著使用除了「rating」以外的「feature」, 並說明你的作法和結果，結果好壞不會影響評分。
* 額外使用的「feature」：年齡、性別、職業、電影年份、電影分類
* 實驗作法
    * 和「MF」相同，也有「User/Movie Embedding」和「User/Movie Bias」
    * 讓「職業」和「電影分類」，也各自有一個「Embedding」(職業使用「Embedding Layer」，而電影分類使用「Dense」)
    * 每筆資料現在都有$4$個「Embedding」，任兩個作內積，共可得到$C^4_2=6$個內積結果
    * 把$6$個內積結果，以及「年齡」、「性別($0$/$1$)」和「電影年份」，「concat」成一個$9$維的向量
    * 過一層「Dense Layer」，拿到一個實數，加上「User/Movie」的「Bias」，就是最後預測的評分
* 實驗結果
    * 「Validation MSE」最低可到$0.74$左右，相較於原始的「MF」，進步了一些

