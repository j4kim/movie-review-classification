# movie-review-classification
Classification de critique de film avec scikit-learn 

## Prérequis

* [scikit-learn](http://scikit-learn.org/stable/install.html)

## Utilisation

### Extraction des données

Décompressez l'archive `tagged.7z`

### Création des corpus

```
python separation.py tagged
```

Où tagged est un dossier contenant deux dossiers : `pos` et `neg` contenant les critiques canonisées.

Ce script va diviser aléatoirement les critiques de chaque catégories en un corpus d'entraîment `train` et un corpus de test `test`. Ces deux corpus sont sauvegardés dans un dossier `tagged_prepared`.

### Classification 

```
python classification.py tagged_prepared
```

Ce script utilise et compare plusieurs méthode de classification. Chaque classificateur utilise les données dans `train` pour s'entraîner. Les données `test` sont ensuite utilisée pour évaluer la justesse du classificateur.
