# 📚 Projet M2 ESI – Machine Learning & Deep Learning

Ce projet est divisé en **3 parties**, chacune explorant une facette du Machine Learning ou du Deep Learning à travers un cas concret :

1. **Reconnaissance d’images avec CNN (TensorFlow + Keras)**
2. **Détection de fraude avec données tabulaires (Scikit-learn)**
3. **Analyse de texte (NLP) avec LSTM (TensorFlow)**

---

## Partie 1 : Reconnaissance d’Images avec les CNN

### 🔍 Objectif
Construire un modèle CNN pour classifier les images du dataset CIFAR-10.

### 📸 Captures d'écran

1. **Affichage d’images du dataset**  
![Capture 1](./images/capture1.png)

2. **Console d'entraînement du modèle**  
![Capture 2](./images/capture2.png)

3. **Courbes d’apprentissage (accuracy / loss)**  
![Capture 3](./images/capture3.png)

4. **Matrice de confusion**  
![Capture 4](./images/capture4.png)

5. **TensorBoard** *(optionnel)*  
![Capture 5](./images/capture5.png)

6. **Tableau de test d’hyperparamètres** *(si implémenté)*  
![Capture 6](./images/capture6.png)

---

### ❓ Questions / Réponses Partie 1

#### 1. Quel a été le rôle des callbacks ModelCheckpoint et EarlyStopping ? Ont-ils été utiles pendant votre entraînement ?
> Les callbacks ont permis de sauvegarder automatiquement le meilleur modèle (`ModelCheckpoint`) et d'arrêter l'entraînement lorsque la performance ne progressait plus (`EarlyStopping`). Cela a évité le sur-apprentissage et a fait gagner du temps.

#### 2. Analysez les courbes d'apprentissage (perte et précision). Votre modèle présente-t-il des signes de sur-apprentissage ou de sous-apprentissage ? Comment la matrice de confusion vous aide-t-elle à comprendre les erreurs du modèle ?
> Le modèle montre de légers signes de sur-apprentissage : la précision d'entraînement continue d'augmenter alors que celle de validation stagne. La matrice de confusion montre que certaines classes sont confondues (ex : chats/chiens), ce qui permet d’identifier les faiblesses du modèle.

#### 3. Quels ont été les principaux défis rencontrés ? Comment pourriez-vous améliorer les performances de ce premier modèle ?
> Le principal défi a été de bien comprendre les couches CNN et d’obtenir un bon équilibre entre performance et simplicité. On pourrait améliorer les performances avec de la **data augmentation**, du **dropout** ou en testant d'autres architectures.

#### 4. Comparez les performances (précision, temps d'entraînement) du modèle simple et du modèle basé sur le transfert d'apprentissage. Pourquoi observe-t-on une telle différence ?
> Le modèle avec transfert d’apprentissage (MobileNetV2) a appris plus vite et a obtenu de meilleurs résultats, car il est pré-entraîné sur un gros dataset (ImageNet). Il capte mieux les formes générales même avec peu de données.

---

## Partie 2 : Données Tabulaires – Détection de Fraude

### 📸 Captures d'écran

1. **Analyse exploratoire des données (EDA)**  
![Capture 7](./images/capture7.png)

2. **Console d'entraînement du modèle RandomForest**  
![Capture 8](./images/capture8.png)

3. **Matrice de confusion + classification report**  
![Capture 9](./images/capture9.png)

---

### ❓ Questions / Réponses Partie 2

#### 1. Expliquez le rôle du paramètre `class_weight='balanced'`. Quelles autres techniques auriez-vous pu utiliser pour gérer le déséquilibre des classes ?
> Ce paramètre donne plus de poids à la classe minoritaire (fraude). Cela force le modèle à accorder plus d’importance à la détection des fraudes. On aurait aussi pu utiliser du **sur-échantillonnage** (SMOTE) ou du **sous-échantillonnage**.

#### 2. Quel type d'apprentissage et quel type de problème est-ce ? Argumentez.
> Il s'agit d’un **apprentissage supervisé**, car on a des étiquettes connues (fraude ou non). C’est un problème de **classification binaire**, car il n'y a que deux classes.

#### 3. Donnez un exemple de régression lié à ce type de données.
> On pourrait imaginer une régression pour **prédire le montant probable d’une transaction future**, ou le **risque associé à un utilisateur** selon son historique.

---

## Partie 3 : NLP – Catégorisation de Dépenses

### 📸 Captures d'écran

1. **Aperçu du dataset texte**  
![Capture 10](./images/capture10.png)

2. **Architecture du modèle LSTM (summary)**  
![Capture 11](./images/capture11.png)

3. **Courbes d’apprentissage NLP**  
![Capture 12](./images/capture12.png)

---

### ❓ Questions / Réponses Partie 3

#### 1. Quel est l’intérêt de la couche `TextVectorization` ?
> Elle transforme le texte brut en séquences numériques exploitables par le réseau de neurones. Cela inclut la tokenisation, la normalisation et la vectorisation.

#### 2. Pourquoi utilise-t-on une couche `Embedding` ?
> Elle permet de convertir chaque mot en un vecteur dense (apprentissage de représentations sémantiques), ce qui aide le modèle à comprendre le sens des mots.

#### 3. Quelle est l'utilité de `Bidirectional(LSTM)` ?
> Elle lit le texte dans les deux sens (gauche → droite et droite → gauche), ce qui permet de mieux capturer le contexte des mots.

---

## 📁 Arborescence du dépôt

```
├── images/
│ ├── capture1.png
│ ├── capture2.png
│ └── ...
├── README.md
└── rapport.pdf
├── partie1_CNN.ipynb
├── partie2_Tabulaires_RF.ipynb
├── partie3_NLP_LSTM.ipynb
```

---

## ✉️ Auteurs
- Jayson, Sofiane, Hamidou, Antony et Ramdhan 
- Promo : Master 2 ESI