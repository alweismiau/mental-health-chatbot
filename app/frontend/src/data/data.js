const questionsEI = [
  {
    question: "Ketika bertemu dengan orang baru kamu biasanya",
    options: [
      { text: "Memulai membuka topik pembicaraan", value: "E" },
      { text: "Menunggu orang membuka topik pembicaraan", value: "I" },
    ],
  },
  {
    question:
      "Pekerjaan yang mengharuskan diri kamu menghadapi banyak orang baru membuat kamu…",
    options: [
      { text: "Lebih semangat bekerja", value: "E" },
      { text: "Merasa tertekan", value: "I" },
    ],
  },
  {
    question: "Mana yang lebih kamu pilih?",
    options: [
      { text: "Banyak teman walaupun jarang bertemu", value: "E" },
      { text: "Sedikit teman, namun sering bertemu", value: "I" },
    ],
  },
  {
    question: "Dalam kelompok pertemanan kamu, kamu adalah seorang…",
    options: [
      { text: "“Badut” yang meramaikan setiap pertemuan", value: "E" },
      {
        text: "“Tempat” curhat karena dianggap merupakan pendengar yang baik",
        value: "I",
      },
    ],
  },
  {
    question: "Ketika ada masalah biasanya kamu…",
    options: [
      { text: "Memilih untuk menceritakannya kepada teman-teman", value: "E" },
      { text: "Memilih untuk memendamnya", value: "I" },
    ],
  },
  {
    question: "Ketika ada gosip baru, kamu…",
    options: [
      { text: "Selalu ingin jadi yang pertama kali tahu", value: "E" },
      { text: "Tidak terlalu peduli", value: "I" },
    ],
  },
  {
    question: "Ketika bertemu orang baru, kamu biasanya…",
    options: [
      {
        text: "Mudah memulai pembicaraan dan pembicaraan tersebut bertahan lama",
        value: "E",
      },
      {
        text: "Sulit mencari topik pembicaraan dan biasanya pembicaraan tidak bertahan lama",
        value: "I",
      },
    ],
  },
  {
    question: "Ketika harus berbicara di depan publik, kamu biasanya…",
    options: [
      { text: "Spontan dengan apa yang ingin dibicarakan", value: "E" },
      {
        text: "Melatih apa yang ingin dibicarakan beberapa kali sebelum acara berlangsung",
        value: "I",
      },
    ],
  },
  {
    question: "Ketika mendatangi pesta, kamu biasanya…",
    options: [
      {
        text: "Bertahan sampai acara selesai karena merasa semakin bersemangat",
        value: "E",
      },
      {
        text: "Memilih untuk pulang lebih awal karena merasa semakin lelah",
        value: "I",
      },
    ],
  },
  {
    question: "Ketika tidak suka pada seseorang, kamu biasanya…",
    options: [
      { text: "Langsung mengomunikasikan kepada orang tersebut", value: "E" },
      {
        text: "Tidak mengutarakan kepada orang tersebut karena merasa diri sendiri memiliki kekurangan",
        value: "I",
      },
    ],
  },
];

const questionsNS = [
  {
    question: "Mana yang lebih kamu percayai?",
    options: [
      { text: "Intuisi", value: "N" },
      { text: "Fakta", value: "S" },
    ],
  },
  {
    question:
      "Jika sekarang kamu diminta untuk menulis buku, judul apa yang akan kamu pilih?",
    options: [
      { text: "“Peluang Bisnis di Masa Depan”", value: "N" },
      { text: "“Cara Praktis Menjadi Kaya dalam Waktu 1 Bulan”", value: "S" },
    ],
  },
  {
    question: "Menurutmu, anak-anak seharusnya…",
    options: [
      {
        text: "Melatih kemampuan imajinasinya agar dapat berinovasi",
        value: "N",
      },
      { text: "Melatih kemampuannya untuk berpikir secara logis", value: "S" },
    ],
  },
  {
    question: "Menurut, orang tidak akan dapat hidup tanpa…",
    options: [
      { text: "Prinsip", value: "N" },
      { text: "Fakta", value: "S" },
    ],
  },
  {
    question:
      "Mana yang lebih menarik bagimu jika kamu bekerja di suatu perusahaan?",
    options: [
      { text: "Berspekulasi terkait peluang untuk produk lainnya", value: "N" },
      {
        text: "Memproduksi suatu barang yang terbukti laris di perusahaan",
        value: "S",
      },
    ],
  },
  {
    question: "Kamu yakin bahwa bungee jumping (terjun lenting) aman dilakukan dari…",
    options: [
      { text: "Firasatmu berkata hal tersebut aman", value: "N" },
      {
        text: "Teman-temanmu, berdasarkan pengalaman mereka mengatakan aman",
        value: "S",
      },
    ],
  },
  {
    question: "Ketika melakukan sesuatu biasanya kamu…",
    options: [
      { text: "Melakukannya dengan caramu sendiri", value: "N" },
      {
        text: "Melakukannya dengan cara yang orang lain lakukan pada umumnya",
        value: "S",
      },
    ],
  },
  {
    question: "Kamu lebih mengagumi orang yang…",
    options: [
      { text: "Dapat mempersuasi lewat visi", value: "N" },
      { text: "Dapat memberikan bukti konkret", value: "S" },
    ],
  },
  {
    question: "Kamu lebih tertarik pada…",
    options: [
      { text: "Apa yang mungkin terjadi nanti", value: "N" },
      { text: "Apa yang sedang terjadi saat ini", value: "S" },
    ],
  },
];

const questionsTF = [
  {
    question:
      "Ketika ada yang menanyakan perihal cinta padamu, kamu akan menjawab bahwa cinta itu…",
    options: [
      { text: "Hubungan timbal-balik yang intim antara dua orang", value: "T" },
      {
        text: "Sebuah ikatan emosional yang mengikat dua orang sehingga menciptakan rasa saling memiliki",
        value: "F",
      },
    ],
  },
  {
    question: "Mana yang menurutmu lebih penting?",
    options: [
      { text: "Ketepatan dalam berpikir", value: "T" },
      { text: "Hubungan antarindividu yang harmonis", value: "F" },
    ],
  },
  {
    question: "Menurutmu, mana yang lebih memuaskan?",
    options: [
      { text: "Mendiskusikan isu secara mendetail", value: "T" },
      { text: "Mencapai persetujuan bersama", value: "F" },
    ],
  },
  {
    question: "Mana yang menurutmu lebih mudah kamu korbankan?",
    options: [
      { text: "Keputusan tepat berdasarkan nalar", value: "T" },
      { text: "Hubungan baik dengan orang lain", value: "F" },
    ],
  },
  {
    question: "Pujian mana yang akan lebih membuatmu merasa terapresiasi?",
    options: [
      { text: "“Kamu orang yang sangat logis, ya”", value: "T" },
      { text: "“Kamu sangat perhatian, deh”", value: "F" },
    ],
  },
  {
    question: "Kamu akan lebih percaya dengan orang yang…",
    options: [
      { text: "Kuat dengan penjelasan nalarnya", value: "T" },
      { text: "Mampu mengekpresikan emosi", value: "F" },
    ],
  },
  {
    question: "Akan lebih buruk bagimu jika menjadi orang yang…",
    options: [
      { text: "Tidak logis", value: "T" },
      { text: "Tidak memiliki belas kasih", value: "F" },
    ],
  },
  {
    question: "Apa yang paling mempengaruhimu?",
    options: [
      { text: "Isi kepalamu", value: "T" },
      { text: "Isi hatimu", value: "F" },
    ],
  },
  {
    question: "Mana yang lebih penting bagimu?",
    options: [
      { text: "Dapat meyakinkan orang dengan kata-katamu", value: "T" },
      { text: "Dapat menyentuh orang dengan kata-katamu", value: "F" },
    ],
  },
  {
    question: "Mana yang paling menggambarkan dirimu?",
    options: [
      { text: "Pemikir kritis", value: "T" },
      { text: "Perasa", value: "F" },
    ],
  },
  {
    question: "Pilih salah satu!",
    options: [
      { text: "Berorientasi pada masalah (problem-oriented)", value: "T" },
      { text: "Berorientasi pada orang (people-oriented)", value: "F" },
    ],
  },
];

const questionsPJ = [
  {
    question: "Kamu merupakan orang yang…",
    options: [
      { text: "Ngaretan", value: "P" },
      { text: "Tepat waktu", value: "J" },
    ],
  },
  {
    question: "Kamu lebih memilih…",
    options: [
      { text: "Fleksibel dengan segala sesuatu", value: "P" },
      { text: "Merencanakan segala sesuatu", value: "J" },
    ],
  },
  {
    question: "Pekerjaan seperti apa yang lebih kamu minati?",
    options: [
      { text: "Fleksibel", value: "P" },
      { text: "Sistematis", value: "J" },
    ],
  },
  {
    question: "Jika ada tugas, kamu biasanya lebih memilih…",
    options: [
      { text: "Mengerjakan ketika sudah mendekati deadline", value: "P" },
      { text: "Nyicil dari jauh-jauh hari", value: "J" },
    ],
  },
  {
    question: "Kamu tipe orang yang…",
    options: [
      { text: "Spontan", value: "P" },
      { text: "Tenang dan berhati-hati", value: "J" },
    ],
  },
  {
    question: "Jika belanja ke swalayan, kamu memilih untuk…",
    options: [
      {
        text: "Membeli barang sesuai dengan apa yang diingat dan terbuka pada kemungkinan pilihan untuk membeli barang lain",
        value: "P",
      },
      {
        text: "Mencatat daftar barang yang perlu dibeli agar tidak membeli barang yang tidak diperlukan",
        value: "J",
      },
    ],
  },
  {
    question: "Pilihlah salah satu!",
    options: [
      {
        text: "Hidup fleksibel dan mudah beradaptasi pada perubahan",
        value: "P",
      },
      { text: "Hidup teratur dan terorganisasi", value: "J" },
    ],
  },
  {
    question: "Ketika memutuskan sesuatu kamu biasanya…",
    options: [
      { text: "Memutuskannya dengan spontan", value: "P" },
      { text: "Memutuskannya dengan sangat hati-hati", value: "J" },
    ],
  },
  {
    question: "Liburan seperti apa yang menurutmu lebih dapat kamu nikmati?",
    options: [
      { text: "Dadakan", value: "P" },
      { text: "Terencana dengan baik", value: "J" },
    ],
  },
  {
    question: "Menurutmu, kamu adalah orang yang…",
    options: [
      { text: "Easy-going / santai", value: "P" },
      { text: "Determined / gigih / memiliki tekad yang besar", value: "J" },
    ],
  },
];

const mbtiDescriptions = {
  ISTJ: "ISTJ adalah sosok yang bertanggung jawab, teliti, dan menghargai struktur serta tradisi. Mereka dikenal dapat diandalkan dan logis dalam menyelesaikan tugas.",
  ISFJ: "ISFJ adalah pribadi yang setia, perhatian, dan penuh empati. Mereka senang membantu orang lain dengan cara yang tenang dan praktis.",
  INFJ: "INFJ adalah individu visioner yang penuh empati dan memiliki komitmen kuat terhadap nilai-nilai. Mereka sering menjadi penasihat atau penggerak perubahan.",
  INTJ: "INTJ adalah pemikir strategis yang mandiri dan analitis. Mereka senang merencanakan jangka panjang dan mencari cara-cara efisien untuk mencapai tujuan.",
  
  ISTP: "ISTP adalah tipe yang logis, praktis, dan suka mengeksplorasi bagaimana sesuatu bekerja. Mereka cenderung tenang tapi suka tantangan teknis dan langsung.",
  ISFP: "ISFP adalah pribadi yang artistik, lembut, dan hidup di saat ini. Mereka sangat menghargai kebebasan dan keindahan dalam hidup.",
  INFP: "INFP adalah idealis yang penuh empati dan imajinatif. Mereka termotivasi oleh nilai-nilai pribadi dan sering ingin membuat dunia menjadi tempat yang lebih baik.",
  INTP: "INTP adalah pemikir logis dan analitis yang senang mengeksplorasi ide abstrak. Mereka sering tertarik pada teori dan solusi inovatif.",
  
  ESTP: "ESTP adalah orang yang energik, spontan, dan penuh aksi. Mereka suka memecahkan masalah secara langsung dan menikmati pengalaman baru.",
  ESFP: "ESFP adalah pribadi yang ceria, hangat, dan sangat sosial. Mereka menikmati momen saat ini dan senang menghibur orang di sekitar mereka.",
  ENFP: "ENFP adalah individu yang penuh semangat, kreatif, dan ekspresif. Mereka sangat peduli dengan orang lain dan ingin mengejar makna hidup.",
  ENTP: "ENTP adalah inovator yang cerdas, penuh rasa ingin tahu, dan suka debat sehat. Mereka senang mengeksplorasi kemungkinan dan berpikir out of the box.",
  
  ESTJ: "ESTJ adalah pemimpin alami yang logis, tegas, dan menghargai keteraturan. Mereka suka mengatur sistem dan memastikan semuanya berjalan efisien.",
  ESFJ: "ESFJ adalah tipe yang ramah, terorganisir, dan peduli pada keharmonisan sosial. Mereka senang menjaga hubungan dan mendukung orang lain.",
  ENFJ: "ENFJ adalah komunikator ulung yang hangat, penuh empati, dan memotivasi. Mereka pandai memimpin dan menginspirasi kelompok menuju tujuan bersama.",
  ENTJ: "ENTJ adalah tipe yang ambisius, percaya diri, dan sangat terarah. Mereka unggul dalam merancang strategi dan mengambil alih kepemimpinan."

};

const MBTIImages = {
  ISTJ: "https://i.pinimg.com/736x/36/5a/15/365a154327362419bbd17449a70b2713.jpg",
  ISFJ: "https://i.pinimg.com/736x/e0/83/32/e0833218cb581ddd5e2ef23b901bc74b.jpg",
  INFJ: "https://i.pinimg.com/736x/fd/07/b4/fd07b471daa477b754d944ab5760bc75.jpg",
  INTJ: "https://i.pinimg.com/736x/4d/b2/5b/4db25b66fb6a8a488d91d6a96827014c.jpg",
  ISTP: "https://i.pinimg.com/736x/4c/e6/62/4ce662c9e1117745bddca5a15f0a4307.jpg",
  ISFP: "https://i.pinimg.com/736x/00/19/1b/00191b490a29d450d67f881e2e27b8de.jpg",
  INFP: "https://i.pinimg.com/736x/ee/6f/ad/ee6fadc3f14d53eeea88559ea8ff1909.jpg",
  INTP: "https://i.pinimg.com/736x/cf/2f/02/cf2f0231b598da88563e6516a0fb5164.jpg",
  ESTP: "https://i.pinimg.com/736x/f1/5d/5e/f15d5eeb77a6f77d6e18a86ca46c9701.jpg",
  ESFP: "https://i.pinimg.com/736x/f3/dc/d9/f3dcd9c94642ffff9c88e52ebaef2355.jpg",
  ENFP: "https://i.pinimg.com/736x/f1/5a/86/f15a86983905e9011bdf24bc30feb631.jpg",
  ENTP: "https://i.pinimg.com/736x/0a/0f/98/0a0f9861bb91712225f7d8f683467582.jpg",
  ESTJ: "https://i.pinimg.com/736x/27/cb/af/27cbaf0967c584c39da44f01a4e94ac8.jpg",
  ESFJ: "https://i.pinimg.com/736x/d1/cd/5e/d1cd5e697e0c5b4265c32d7a2b7f54a4.jpg",
  ENFJ: "https://i.pinimg.com/736x/1e/39/bf/1e39bf3d1919f70d646073d07515bf24.jpg",
  ENTJ: "https://i.pinimg.com/736x/41/fb/e4/41fbe49be6e1f9efa19ce26c16a97911.jpg"
};


export { questionsEI, questionsNS, questionsTF, questionsPJ, mbtiDescriptions, MBTIImages };
