# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [2.0.6] 2023-01-26
### Fixed
- no time to explain

## [2.0.5]
### Changed

- Documentation update.

## [2.0.4]
### Fixed

- Plugin naming fix (build.sbt/plugin.conf).

## [2.0.3]
### Changed

- В XgboostClassifier/XgboostRegressor трекеры изменены с питоновских на скаловые.

## [2.0.1]
### Changed

- Документация переведена на движок `mkdocs` и полностью обновлена
- Добавлен класс MLCoreModelSpec для тестов, в нем исправлена проверка 
  совпадения схем датафреймов (без учета nullable)
- Во многих алгоритмах исправлены типы полей в тестовых датафреймах (убраны 
  кавычки вокруг чисел при задании датафреймов)
- Убрано создание ненужных полей в команде `sample`.
- Убрано приведение названия класса к типу Double при сэмплинге (`feature.
  Sampling`)

## [2.0.0]

- Изменен package на com.isgneuro.otp.plugins.mlcore
- Команда pivot перенесена в OTLExtend.

## [1.2.3]
### Added

- Команды OTL: sample, pivot, makefuture, describe
- Алгоритмы SMaLL: 
  - apply: bucket (QuantileDiscretizer)
  - fit: CountVectorizer, TF-IDF, TextClustering, DecisionTreeClassifier  
- добавлены гиперпараметры l2LeafReg и borderCount для катбуста

### Changed

- В команду FillNA добавлена поддержка by
- В Correlation добавлен параметр, какой тип таблицы выводить (wide|long, default=wide)
- в тестах проверяются все возможные гиперпараметры, также где возможно вынес общий для датасет на уровень выше отдельных тестов (задел на то, чтобы потом унести в plugin-mlcore)

### Fixed

- исправлены результирующие поля:
  - регрессия теперь везде возвращает <modelname>_prediction
  - классификация возвращает <modelname>_prediction и probabilities с вероятностями через запятую
- убраны мусорные поля (кроме пустого поля с названием алгоритма), кажется баг вне plugin-mlcore


## [1.0.0] - 2020-11-24
### Added

- This changelog
- Implementation of the following ML algorithms as extensions for SMaLL commands:
  - Regression: linear regression, random forest, gradient boosting
  - Classification: logistic regression, random forest, gradient boosting
  - Clustering: KMeans, DBScan
  - Anomaly detection: isolation forest, local outlier factor, z-score, IQR and MAD statistics
  - Feature engineering: PCA
  - Stats functions: Pearson and Spearman correlation
  - Time series: prediction with STL-decomposition and linear regression.