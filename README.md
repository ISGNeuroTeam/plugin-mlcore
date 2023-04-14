# plugin-mlcore

<div align="center">
   <p align="center"> <img src="mlcore-image.jpg" height=240px; weight=320px;"><br></p>
</div>

*MLCore is a plugin for OT.Platform. MLCore delivers new **SMaLL extensions** which implement core ML concepts and algorithms.*

---

## Installation

1. make pack or make build.
2. MLCore delivered as JAR-file named `plugin-mlcore_<SCALAVERSION>-<PLUGINVERSION>.jar` (ex.: plugin-mlcore_2.11-2.0.0.jar). Put this file into `/opt/otp/dispatcher/plugins/plugin-mlcore/`.
3. External dependencies for MLCore delivered in `lib` folder. Copy this folder into `/opt/otp/dispatcher/plugins/plugin-mlcore/`.
4. Make sure that your OT.Platform installation includes `smallplugin-core` and `smallplugin-sdk`:    
  - Check `/opt/otp/dispatcher/plugins/smallplugin-core` path. It should contain two jar-files: `smallplugin-core*.jar` and `smallplugin-sdk*.jar`. If absent, create a folder and put these JARs there.
4. Make sure that the following config files are located in `smallplugin-core` folder. If absent, copy these files there.
  - plugin.conf
  - loglevel.properties  
5. Restart dispatcher.

## Dependencies

- smallplugin-sdk_2.11  0.3.0
- sbt 1.5.8
- scala 2.11.12
- eclipse temurin 1.8.0_312 (ранее назывался AdoptOpenJDK)

## Usage

Use MLCore algorithms as extensions for `fit` and `apply` SMaLL-commands.

```
| inputlookup realEstate.csv
| apply dbscan from longitude, latitude
```

## Documentation

You can find actual documentation [here].

## Roadmap

- Dashboards or Zeppelin notebooks with end-to-end examples
- Extensions for `score` SMaLL command
- Model summary
- One-Hot Encoding
- Cross-validation, train/test split
- Statistical tests
- Extensions for `explain` SMaLL-command (model interpretation)
- Human-like aliases for default algorithms 

## Authors

 
 
