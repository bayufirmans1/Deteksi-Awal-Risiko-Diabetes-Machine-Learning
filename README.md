
# Deteksi Awal Risiko Diabetes Menggunakan Machine Learning
###### 


## Domain Proyek
### Latar Belakang
Diabetes adalah penyakit kronis dan metabolik yang ditandai dengan tingginya kadar glukosa darah atau gula darah, yang seiring waktu dapat menyebabkan kerusakan serius pada jantung, pembuluh darah, mata, ginjal, dan saraf. Kondisi kronis ini terjadi baik ketika pankreas tidak menghasilkan cukup insulin (hormon yang mengatur gula darah, atau glukosa), atau ketika tubuh tidak dapat secara efektif menggunakan insulin yang dihasilkan. 
Menurut *World Health Organization* (WHO) sekitar 422 juta orang di seluruh dunia memiliki diabetes, dengan mayoritas tinggal di negara-negara berpendapatan rendah dan menengah, dan 1,5 juta kematian langsung disebabkan oleh diabetes setiap tahunnya. Baik jumlah kasus maupun prevalensi diabetes terus meningkat selama beberapa dekade terakhir.

Berdasarkan *International Diabetes Federation* (IDF) dalam Atlas edisi ke-10, Pada tahun 2021, lebih dari lebih dari setengah miliar manusia dari seluruh dunia hidup dengan diabetes, atau tepatnya 537 juta orang pada kelompok orang dewasa berusia antara 20–79 tahun. Indonesia menempati peringkat kelima populasi diabetes dewasa terbanyak diperkirakan sebanyak 19,5 juta orang atau prevalensi diabetes pada usia antara 20-79 tahun adalah 10,6%. Indonesia juga termasuk peringkat terbanyak pengidap diabetes yang belum terdiagnosa pada usia antara 20–79 tahun diperkirakan sebanyak 14,3 juta orang atau 73,7% proporsi yang belum terdiagnosa.[[3]](https://diabetesatlas.org/atlas/tenth-edition/)

Mengingat keterkaitan risiko berkembangnya komplikasi dari penyakit diabetes yang dapat menyebabkan kerusakan serius pada organ tubuh hingga efek terburuk yaitu kematian, maka deteksi dini penyakit diabetes menjadi penting untuk dilakukan. Semakin dini diagnosis dan deteksi dilakukan, semakin mudah penyakit diabetes dikontrol dan diobati. 

Dengan    perkembangan    teknologi,    diabetes    dapat    di identifikasi  sejak  dini  dengan  menggunakan  pendekatan Data  mining yaitu proses  pengumpulan dan  pengolahan  data  yang  bertujuan  untuk  mengekstrak informasi  penting  atau  sebuah  pola  pada data.  Proses ekstraksi   dapat   dilakukan   dengan algoritma *machine learning*.

Penelitian mengenai implementasi algoritma *machine learning* untuk mengidentifikasi penderita diabetes telah dilakukan sebelumnya beberapa diantaranya yaitu pada penelitian Sisodia et al. (2018) dengan judul *Prediction of Diabetes using Classification Algorithms* melakukan prediksi penyakit diabetes menggunakan algoritma *Decision Tree*, *Support Vector Machines* dan *Naive Bayes*, menghasilkan *Naive Bayes* dengan akurasi tertinggi dibanding  algoritma lainnya yaitu akurasi sebesar 76,3%.[[4]](https://doi.org/10.1016/j.procs.2018.05.122). Pada penelitian lain Adigun et al. (2022) melakukan penelitian dengan topik serupa dengan judul *Classification of Diabetes Types using Machine Learning* menggunakan algoritma *Decision Tree*, *Random Forest*, dan *Support Vector Machines*, menghasilkan akurasi tertinggi menggunakan *Random Forest* sebesar 100%.[[1]](https://www.researchgate.net/publication/364055325_Classification_of_Diabetes_Types_using_Machine_Learning)

Pada proyek ini akan menggunakan algoritma *machine learning* untuk membuat model yang dapat melakukan prediksi diagnosa diabetes atau tidak, sehingga dapat digunakan sebagai acuan untuk pengobatan penderita diabetes bagi dokter di rumah sakit dan di masyarakat untuk mengetahui cara menjaga pola hidup dan cara menghindari penyakit diabetes dilihat dari variabel yang mempengaruhi terjadinya penyakit.

## Business Understanding
Penyakit diabetes seiring waktu dapat menyebabkan kerusakan serius pada jantung, pembuluh darah, mata, ginjal, dan saraf hingga efek terburuknya yaitu kematian, maka deteksi dini penyakit diabetes menjadi penting untuk dilakukan. Ketika terdeteksi secara dini, pasien tidak hanya dapat menunda, bahkan juga dapat mencegah perkembangan penyakit menjadi diabetes akut. Pencegahan penyakit secara signifikan lebih murah dan mudah daripada pengobatan hiperglikemia dan komplikasi diabetes. Oleh sebab itu, cara identifikasi, diagnosis, dan analisis diabetes secara cepat dan akurat merupakan topik proyek yang sangat bermanfaat dan penting untuk dilakukan. Dalam bidang kedokteran, diagnosis penyakit diabetes dilakukan berdasarkan kadar gula darah, di antaranya kadar gula darah sewaktu, kadar gula darah puasa, dan kadar toleransi gula darah [[2]](https://doi.org/10.2337/dc22-S002). Hasil pengukuran kadar gula darah ini akan menunjukkan seseorang menderita diabetes atau tidak. Semakin dini diagnosis dan deteksi dilakukan, semakin mudah penyakit diabetes dikontrol dan diobati.

Cara lain untuk mendeteksi penyakit diabetes adalah
dengan memanfaatkan algoritma *machine learning* dengan klasifikasi. klasifikasi adalah metode dalam *machine learning* yang digunakan oleh mesin untuk memilah atau untuk mengklasifikasikan obyek berdasarkan ciri tertentu sebagaimana manusia mencoba membedakan benda satu dengan yang lain, klasifikasi bertujuan untuk memisahkan data menjadi kelas-kelas tertentu.
Pada proyek ini akan membangun model _machine learning_ yang dapat memprediksi diagnosa penyakit diabetes berdasarkan pengukuran jumlah kehamilan, konsentrasi glukosa plasma, Tekanan darah diastolik, Ketebalan lipatan kulit trisep, Insulin, indes massa tubuh, DiabetesPedigreeFunction, dan umur, data yang digunakan adalah Pima Indians Diabetes Database.

### Problem Statement
- Bagaimana cara melakukan *preprocessing* data agar memiliki model yang terbaik untuk melakukan deteksi awal risiko diabetes?
- Bagaimana cara membangun model _machine learning_ untuk melakukan deteksi awal risiko diabetes?
 
### Goals
- Melakukan *preprocessing* data yang baik sehingga dihasilkan model yang terbaik.
- Membuat model *machine learning* yang dapat melakukan deteksi awal risiko diabetes seakurat mungkin.

### Solution Statement
- Memahami data dengan cara melakukan EDA untuk menemukan pola, menemukan anomali, mengetahui kolerasi antar fitur dengan bantuan statistik ringkasan dan representasi grafis.
- Mengolah data agar siap digunakan dalam membangun model *machine learning preprocessing* yang akan dilakukan adalah melakukan seleksi fitur, mengidentifikasi dan mengatasi *missing value*, mengatasi *outlier* dengan metode IQR (Interquartile Range), melakukan train test split dengan membagi data menjadi dua dengan data training sebesar 85% dan data testing 15%, dan melakukan standardisasi data untuk membuat beberapa variabel memiliki rentang nilai yang sama.
- Melakukan pelatihan model menggunakan algoritma *Random Forest*,  *Decision Tree* dan *K-Nearest Neighbor* dari library Scikitlearn dan melakukan evaluasi model menggunakan metrik *Accuracy Score* dan *F1 Score*.


## Data Understanding
Dataset yang digunakan dalam proyek ini merupakan data yang diperoleh dari UCI Machine Learning. dataset ini merupakan pengukuran diabetes perempuan berusia diatas 21 tahun dari Suku Indian Pima yang merupakan sekelompok penduduk asli Amerika yang tinggal di daerah yang terdiri dari Arizona tengah dan selatan sekarang, serta Meksiko barat laut di negara bagian Sonora dan Chihuahua. Dataset ini dapat diunduh di [Pima Indians Diabetes Database](thttps://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database).
Dataset ini memiliki 768 baris data pengukuran diagnostik diabetes dan memiliki 9 kolom yaitu `Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age, Outcome`. Dataset ini berisi data numerik yang memiliki 2 jenis tipe data yaitu `BMI, DiabetesPedigreeFunction` bertipe float64 dan kolom lainnya bertipe int64. 
  
 Untuk penjelasan mengenai variabel-variabel pada dataset sebagi berikut.

 - Pregnancies : Berapa kali hamil
 - Glucose : Konsentrasi glukosa plasma lebih dari 2 jam dalam tes toleransi glukosa oral
 - BloodPressure : Tekanan darah diastolik (mm Hg)
 - SkinThickness : Ketebalan lipatan kulit trisep (mm)
- Insulin : Insulin serum 2 jam (mu U/ml)
- BMI : Indeks massa tubuh (berat dalam kg/(tinggi dalam m)^2)
- DiabetesPedigreeFunction : Fungsi silsilah diabetes (fungsi yang menilai kemungkinan diabetes berdasarkan riwayat keluarga)
- Age : Usia (tahun)
- Outcome : Kelas variabel (0 jika non-diabetes, 1 jika diabetes)

**Tabel.1 Deskripsi Data**

|   | count | mean | std | min  | 25%  | 50% | 70% | max  |
|---|--------|-----------|-----------|--------|--------|-----------|----------|--------|
 | Pregnancies	 | 768.0 | 3.845052 | 3.369578 | 0.000 | 1.00000 | 3.0000 | 6.00000 | 17.00 | 
 | Glucose	 | 768.0 | 120.894531 | 31.972618 | 0.000 | 99.00000 | 117.0000 | 140.25000 | 199.00 | 
 | BloodPressure | 768.0 | 69.105469 | 19.355807 | 0.000 | 62.00000 | 72.0000 | 80.00000 | 122.00 | 
 | SkinThickness | 768.0 | 20.536458 | 15.952218 | 0.000 | 0.00000 | 23.0000 | 32.00000 | 99.00 | 
 | Insulin	 | 768.0 | 79.799479 | 115.244002 | 0.000 | 0.00000 | 30.5000 | 127.25000 | 846.00 | 
 | BMI		 | 768.0 | 31.992578 | 7.884160 | 0.000 | 27.30000 | 32.0000 | 36.60000 | 67.10 | 
 | DiabetesPedigreeFunction | 768.0 | 0.471876 | 0.331329 | 0.078 | 0.24375 | 0.3725 | 0.62625 | 2.42 | 
 | Age		 | 768.0 | 33.240885 | 11.760232 | 21.000 | 24.0000 | 29.0000 | 41.00000 | 81.00 | 
 | Outcome	 | 768.0 | 0.348958 | 0.476951 | 0.000 | 0.00000 | 0.0000 | 1.00000 | 1.00 | 

Berdasarkan tabel deskripsi data diatas pada variabel Glucose, BloodPressure, SkinThickness, Insulin, BMI tidak mungkin bernilai nol sehingga terdapat missing value di dataset. 


### Univariate Analysis
Histogram berikut merupakan distribusi data masing-masing variabel.
![Gambar1](https://i.ibb.co/KGX000h/univariate.png)
**Gambar.1**

Pada gambar.1 diatas dapat diketahui bahwa terdapat orang juga memiliki kadar glukosa antara 140 mg/dL sampai 199 mg/dL dan dianggap sebagai penderita pradiabetes. Sebagian besar orang memiliki tekanan darah antara 50-100 mmHg, sebagian besar orang memiliki insulin 0, Nilai BMI berkisar di antara 20 sampai 50, sementara untuk orang dewasa yang sehat harus memiliki BMI antara 18,5-24,9, hal ini memperlihatkan banyak orang yang kelebihan berat badan atau obesitas serta sejumlah besar orang memiliki usia antara 20-40 tahun. Pada distribusi semua variabel mengalami skewness sehingga diperlukan proses normalisasi sebelum digunakan untuk proses pemodelan.


### Multivariate Analysis

Berikut grafik heatmap untuk melihat kolerasi antara semua variabel numerik.

![Gambar2](https://i.ibb.co/7XWJZtt/multivariate-heatmap.png)
**Gambar 2**

Dari gambar.2 grafik heatmap diatas menunjukkan koefisien korelasi antar himpunan variabel. Jika nilai korelasi > 0 maka terdapat korelasi positif. Sementara nilai satu variabel meningkat, nilai variabel lainnya juga meningkat. Jika persamaan korelasi = 0 maka tidak ada korelasi. Jika korelasi < 0 maka ada korelasi negatif. Setiap variabel independen dalam tabel berkorelasi dengan masing-masing nilai lain dalam tabel. Dengan demikian, semua variabel digunakan untuk modelling. Korelasi yang cukup kuat antara variabel Glucose, BMI, Age, dan Pregnancies dengan variabel Target.


## Data Preparation

Proses data preparation adalah tahapan yang dilakukan sebelum melakukan analisis atau pemodelan pada data. Tahapan ini melibatkan pengolahan dan transformasi data agar siap digunakan dalam proses selanjutnya. Berikut adalah beberapa langkah yang dilakukan dalam proses data preparation 

Pada tabel.2 presentase missing value tertinggi yaitu pada fitur Insulin dan fitur Pregnancies, DiabetesPedigreeFunction dan Age tidak terdapat missing value.

- **Seleksi Fitur**
Menyeleksi fitur dilakukan dengan membandingkan korelasi fitur terhadap target dan nilai presentase missing value, missing value lebih dari 20% tidak akan digunakan dalam model. Fitur yang tidak digunakan adalah SkinThickness dengan presentase mising value 29,5% dan fitur Insulin dengan presentase 48,7%.  

- **Penanganan Missing Value**

**Tabel.3 Missing Value**
| Variabel | Jumlah missing value |
|--|--|
| Pregnancies | 0 |
| Glucose | 5 |
| BloodPressure | 35 |
| BMI | 11 |
| DiabetesPedigreeFunction | 0 |
| Age | 0 | 
| Outcome | 0 |

Terdapat 56 missing value dari 3 variabel, untuk mengatasi data yang bernilai nol tersebut data diubah menjadi NaN. Berdasarkan distribusi data ketiga variabel miring ke kanan, maka data selanjutnya diisi dengan nilai median masing-masing variabel.

- **Penanganan Outlier**
Metode IQR outlier dapat dilakukan dengan menghitung quartile 1 (Q1), quartile 3 (Q3), dan interquartile range (IQR). Kemudian, outlier dapat diidentifikasi dan dihapus dengan menghapus semua data yang berada di luar rentang Q1 - 1,5 x IQR dan Q3 + 1,5 x IQR. Jumlah data outlier yang dihapus sebanyak 61 data.

- **Train Test Split**
Tahap ini melibatkan membagi dataset menjadi dua bagian, yaitu set pelatihan (training set) dan set pengujian (testing set). Set pelatihan digunakan untuk melatih model, sedangkan set pengujian digunakan untuk mengevaluasi kinerja model pada data yang tidak terlihat. Pada proyek ini dataset sebesar 707 dibagi menjadi 600 untuk data latih dan 107 untuk data pengujian. 
Dengan rasio 85% data pada data latih dan 15% pada data pengujian.
Dengan melakukan train test split, kita dapat memastikan bahwa model tidak melihat data pengujian sebelumnya dan evaluasi kinerja model tidak terpengaruh oleh bias.

- **Normalization**
Dengan melakukan normalisasi ini, variabel-variabel dalam dataset akan memiliki rentang nilai yang serupa dan skala yang sama, sehingga algoritma pembelajaran mesin dapat bekerja dengan lebih baik dan hasil analisis atau pemodelan dapat diinterpretasikan dengan lebih mudah. 
Metode yang digunakan di proyek ini adalah Standardisasi yang dilakukan dengan mengubah distribusi nilai variabel sehingga rata-rata nilai yang diamati menjadi 0 dan standar deviasi menjadi 1, menggunakan [StandardScaler](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html) dari library Scikitlearn.



## Modeling
Pada bagian modeling dilakukan pemodelan dengan algoritma *Random Forest*, *Decision Tree* dan *K-Nearest Neighbor*, kemudian melakukan training model dengan data training, dilanjut pengujian model menggunakan metrik evaluasi  accuracy dan f1-score dari library Scikitlearn.

**1. Random Forest**
Random Forest merupakan modifikasi dari bagging. Hanya saja, pada Random Forest terdapat penambahan pada random sub sampling atau pemilihan m variabel yang digunakan dalam membangun pohon. Pembuatan model dilakukan dengan menggunakan modul [RandomForestClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html) dari library Scikitlearn dengan parameter n_estimator = 100 yaitu jumlah trees di forest, max_depth = 16 yaitu kedalaman atau panjang pohon, dan random_state = 12345 digunakan untuk mengontrol random number generator yang digunakan. n_job=-1 berarti menggunakan semua prosesor untuk melakukan pekerjaan yang akan dijalankan secara paralel untuk menyesuaikan dan memprediksi, parameter yang lain diset default.

**Berikut merupakan langkah-langkah dalam klasifikasi random forest**
- Buat Decision Tree-nya dari K data yang sudah dipilih sebelumnya.
- Pilih jumlah N-tree (kumpulan pohon-pohon) yang ingin dibuat. Selanjutnya ulangi langkah 1 dan 2. Intinya terus membuat *decision tree* sebanyak-banyaknya.
- Untuk dataset yang baru, buat setiap N-tree memprediksi kelompok dari dataset yang baru. Kemudian dataset yang baru akan masuk ke kelompok yang memiliki probabilitas tertinggi dari semua kombinasi N-tree.

**Berikut adalah kelebihan dari algoritma Random Forest:**
- Kuat terhadap data outlier (pencilan data).
- Bekerja dengan baik dengan data non-linear.
- Risiko overfitting lebih rendah.
- Berjalan secara efisien pada kumpulan data yang besar.
- Akurasi yang lebih baik daripada algoritma klasifikasi lainnya.

**Berikut adalah kelemahan algoritma Random Forest:**
- Random Forest cenderung bias saat berhadapan dengan variabel kategorik.
- Waktu komputasi pada dataset berskala besar relatif lambat.
- Tidak cocok untuk metode linier dengan banyak fitur sparse.


**2. Decision Tree**
Algoritma ini menggunakan node dan internode untuk prediksi dan klasifikasi. Root node mengklasifikasikan instance dengan fitur yang berbeda. Root node dapat memiliki dua atau lebih cabang, sedangkan leaf node merepresentasikan klasifikasi. Dalam setiap tahap, Pohon keputusan memilih setiap node dengan mengevaluasi perolehan informasi tertinggi di antara semua atribut.[[4]](https://doi.org/10.1016/j.procs.2018.05.122)

Pembuatan model dilakukan dengan menggunakan modul [DecisionTreeClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html) dari library Scikitlearn dengan parameter default, max_depth=32 yaitu kedalaman maksimal pohon, dan random_state = 12345 digunakan untuk mengontrol random number generator yang digunakan.

**Berikut langkah-langkah dari algoritma Decision tree** 
- Mulai dari simpul akar, kita misalkan sebagai S, yang berisi dataset lengkap.
- Ambil atribut terbaik dalam dataset menggunakan *Attribute Selection Measure* (ASM). ASM yang bisa digunakan di antaranya Information Gain dan Gini Index
- Pisahkan himpunan S menjadi himpunan bagian yang berisi kemungkinan nilai untuk atribut terbaik.
- Buat simpul decision tree yang berisi atribut terbaik.
- Buat simpul decision tree baru secara rekursif menggunakan himpunan bagian dari kumpulan data yang dibuat pada langkah ketiga. Lanjutkan proses ini sampai tahap terakhir di mana kita tidak dapat mengklasifikasikan simpul lebih lanjut. Simpul ini yang menjadi simpul akhir atau disebut sebagai simpul daun (leaf node).

**Berikut kelebihan merupakan dari algoritma Decision Tree**
- Mudah dibaca dan ditafsirkan tanpa perlu pengetahuan statistik yang dalam.
- Mudah disiapkan tanpa harus menghitung dengan perhitungan yang rumit
- Proses data *cleaning* cenderung lebih sedikit, kasus nilai yang hilang dan outlier kurang signifikan pada data *decision tree*

**Berikut kekurangan merupakan dari algoritma Decision Tree**
- Sifat tidak stabil, ini menjadi salah satu keterbatasan dari algoritma *decision tree* ketika terdapat perubahan kecil pada data dapat menghasilkan perubahan besar dalam struktur pohon keputusan.
- Dapat terjadi masalah *overfitting*.
- Kurang efektif dalam memprediksi hasil dari variabel kontinu.


**3. K-Nearest Neighbor**
Algoritma K-Nearest Neighbor adalah sebuah metode untuk melakukan klasifikasi terhadap objek yang berdasarkan dari data pembelajaran yang jaraknya paling dekat dengan objek tersebut. KNN merupakan algoritma supervised learning dimana hasil dari *query instance* yang baru diklasifikan berdasarkan mayoritas dari kategori pada algoritma KNN. Dimana kelas yang paling banyak muncul yang nantinya akan menjadi kelas hasil dari klasifikasi. Kedekatan didefinisikan dalam jarak metrik, seperti jarak Euclidean. 

Pembuatan model dilakukan dengan menggunakan modul [KNeighborsClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html) dari library Scikitlearn dengan parameter n_neighbors = 3 yaitu model ini akan membandingkan jarak satu sampel data ke 3 sampel data tetangganya yang terdekat dan metrik yang digunakan untuk mengukur jarak adalah Euclidean, parameter yang lain dipilih secara default.

**Berikut langkah-langkah dari algoritma K-Nearest Neighbor**
- Menentukan parameter 𝐾 (jumlah tetangga paling dekat). 
- Menghitung kuadrat jarak Euclid (*queri instance*) masing-masing objek terhadap data sampel. 
- Kemudian mengurutkan objek-objek tersebut ke dalam kelompok yang mempunyai jarak Euclid terkecil. 
- Mengumpulkan kategori 𝑌 (Klasifikasi *Nearest Neighbor*).
- Dengan menggunakan kategori *Nearest Neighbor* yang paling mayoritas maka dapat diprediksi nilai query instance yang telah dihitung.


**Berikut kelebihan merupakan dari algoritma K-Nearest Neighbor**
- Pelatihan sangat cepat
- Sederhana dan mudah dipelajari
- Tahan terhadap data training yang  _noisy_
- Efektif apabila data trainingnya besar.

**Berikut kekurangan merupakan dari algoritma K-Nearest Neighbor**
- Perlu menentukan nilai dari parameter K (jumlah dari tetangga terdekat).
- Tidak menangani nilai hilang (_missing value_) secara implisit.
- Sensitif terhadap data pencilan (_outlier_).
- Rentan terhadap variabel yang non-informatif.
- Rentan terhadap dimensionalitas yang tinggi.
- Pembelajaran berdasarkan jarak tidak jelas mengenai jenis jarak apa yang harus digunakan dan atribut mana yang harus digunakan untuk mendapatkan hasil yang terbaik.
- Biaya komputasi cukup tinggi karena diperlukan perhitungan jarak dari setiap sampel uji pada keseluruhan sampel training.


## Evaluation

**Accuracy score** adalah metrik evaluasi yang mengukur seberapa sering model melakukan prediksi yang benar dari seluruh prediksi yang dilakukan di dataset. *Accuracy score* cocok digunakan pada dataset yang seimbang, di mana jumlah sampel pada setiap kelas relatif sama. Pada dataset yang tidak seimbang, *accuracy score* dapat memberikan hasil yang menyesatkan karena model dapat mencapai tingkat akurasi yang tinggi hanya dengan memprediksi mayoritas kelas yang dominan. Nilai *accuracy score* berkisar antara 0 dan 1.

Rumus untuk menghitung *accuracy score* adalah sebagai berikut:  

$$Accuracy = \frac{True Positive + True Negative}{(Total Sample Size)}$$

Dalam rumus tersebut, *True Positive* (TP) adalah jumlah prediksi yang benar positif, *True Negative* (TN) adalah jumlah prediksi yang benar negatif, dan *Total Sample Size* adalah jumlah total sampel yang dievaluasi.


**F1 score** adalah metrik evaluasi alternatif yang mengukur kemampuan prediksi model dengan memperhatikan kinerja kelas secara terpisah, bukan secara keseluruhan seperti yang dilakukan oleh accuracy score. *F1 score* menggabungkan dua metrik yang bersaing, yaitu *precision* dan *recall*, untuk menghasilkan skor yang lebih baik. *F1 score* cocok digunakan pada dataset yang tidak seimbang, di mana jumlah sampel pada setiap kelas tidak sama.

$$F1 score=2×\frac{precision×recall}{precision+recall}​$$

Dalam rumus tersebut, *precision* adalah rasio prediksi positif yang benar terhadap total prediksi positif, dan *recall* adalah rasio prediksi positif yang benar terhadap total sampel yang sebenarnya positif. F1 score menggabungkan *precision* dan *recall* menggunakan rata-rata harmonik, sehingga memberikan skor yang lebih baik daripada hanya menggunakan *precision* atau *recall* saja.


**Tabel.3 Evaluasi Model**
|     Algoritma     | Accuracy Score Training | Accuracy Score Testing | F1 Score Training | F1 Score Testing |
|------------------ |---------------| -------- |---------------| -------- |
|   Random Forest   | 1.000000 |  0.841121 | 1.000000 | 0.773333 |
|   Decision Tree   | 1.000000 |  0.719626 | 1.000000 | 0.594595 |
| K-Nearest Neighbor| 0.851667 | 0.785047 | 0.757493 | 0.693333 |

Setelah melakukan evaluasi data training dan data test, berdasarkan tabel 3 diatas model terbaik adalah model *random forest*. Model *random forest* menghasilkan pengujian paling baik diantara ketiga model dengan *Accuracy Score* data testing sebesar 84,11% dan F1 Score 77,33%. 

Berikut grafik visualisasi Accuracy dan F1 Score pada Gambar 3 : 

![Gambar3](https://i.ibb.co/JnSTFvn/eval.png)
**Gambar 3** 

Hasil *Accuracy Score* sebesar 84,11% dan F1 Score 77,33% pada pengujian menggunakan algoritma *random forest* ini lebih tinggi dibanding penelitian Sisodia et al. (2018) yang menghasilkan *Naive Bayes* sebagai model terbaik dengan skor akurasi sebesar 76,3%. Namun masih dibawah penelitian Adigun et al. (2022) yang menghasilkan skor akurasi tertinggi menggunakan *Random Forest* sebesar 100%.


### Kesimpulan

Pada proyek ini disimpulkan bahwa model *Random Forest* merupakan model terbaik dengan nilai *accuracy score* dan *F1 Score* pada pengujian data training dan test tertinggi, model *Decision Tree* pada urutan kedua, dan *K-Nearest Neighbor* pada urutan terakhir.
Berdasarkan hasil proyek prediksi diabetes yang dilakukan dengan menggunakan algoritma *Random Forest*, *Decision Tree* dan *K-Nearest Neighbor*, proyek ini dapat membantu dalam deteksi dini penyakit diabetes dan memberikan prediksi yang akurat untuk pasien diabetes serta dapat digunakan sebagai dasar untuk pengembangan model prediksi yang lebih baik di masa depan.
Pengembangan selanjutnya dapat dilakukan dengan penggunaan hyperparameter tuning untuk meningkatkan kinerja model atau mencoba algoritma klasifikasi lainnya.


## Referensi 
[[1]](https://www.researchgate.net/publication/364055325_Classification_of_Diabetes_Types_using_Machine_Learning) Adigun, Oyeranmi & Folasade, Okikiola & Yekini, Nureni & Babatunde, Ronke. (2022). Classification of Diabetes Types using Machine Learning. International Journal of Advanced Computer Science and Applications. 13.

[[2]](https://doi.org/10.2337/dc22-S002) American Diabetes Association Professional Practice Committee (2022). 2. Classification and Diagnosis of Diabetes: Standards of Medical Care in Diabetes-2022. _Diabetes care_, _45_(Suppl 1), S17–S38. 

[[3]](https://diabetesatlas.org/atlas/tenth-edition/)  International Diabetes Federation. IDF Diabetes Atlas, 10th edn. Brussels, Belgium: 2021. Available at: https://diabetesatlas.org/atlas/tenth-edition/

[[4]](https://doi.org/10.1016/j.procs.2018.05.122) Sisodia, D., & Sisodia, D. S. (2018). Prediction of diabetes using classification algorithms. _Procedia computer science_, _132_, 1578-1585.
