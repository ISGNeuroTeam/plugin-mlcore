
name := "plugin-mlcore"

description := "Plugin for ml algorithms"

version := "2.0.6"

scalaVersion := "2.11.12"

val dependencies = new {
  private val smallPluginSdkVersion = "0.3.0"
  private val sparkVersion = "2.4.3"

  val smallPluginSdk = "ot.dispatcher.plugins.small" % "smallplugin-sdk_2.11" % smallPluginSdkVersion % Compile
  val sparkMlLib = "org.apache.spark" %% "spark-mllib" % sparkVersion % Compile
  val xgboost4jSpark = "ml.dmlc" % "xgboost4j-spark" % "0.90"
  val xgboost4j = "ml.dmlc" % "xgboost4j" % "0.90"

}

// fix for fasterxml dependency conflict with catboost
dependencyOverrides ++= {
  Seq(
    "com.fasterxml.jackson.module" %% "jackson-module-scala" % "2.6.7.1",
    "com.fasterxml.jackson.core" % "jackson-databind" % "2.6.7",
    "com.fasterxml.jackson.core" % "jackson-core" % "2.6.7"
  )
}

libraryDependencies ++= Seq(
  dependencies.smallPluginSdk,
  dependencies.sparkMlLib,
  dependencies.xgboost4jSpark,
  dependencies.xgboost4j
)

Test / parallelExecution := false
