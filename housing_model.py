import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from pathlib import Path
import logging
import joblib
from datetime import datetime
from sklearn.model_selection import StratifiedShuffleSplit, cross_val_score, GridSearchCV
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, ElasticNet
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.inspection import permutation_importance
import warnings
warnings.filterwarnings("ignore")

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("housing_model.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("HousingPriceModel")

class HousingPriceModel:
    """
    Clase para la predicción de precios de viviendas en California.
    
    Implementa un flujo de trabajo completo:
    - Carga y exploración de datos
    - Visualización y análisis
    - Preprocesamiento y feature engineering
    - Entrenamiento y evaluación de modelos
    - Almacenamiento del modelo final
    """
    
    def __init__(self, data_path=None, random_state=42):
        """
        Inicializa el modelo con la ruta a los datos y un estado aleatorio para reproducibilidad.
        
        Parameters:
        -----------
        data_path : str
            Ruta al directorio que contiene los datos de vivienda
        random_state : int
            Semilla para reproducibilidad
        """
        self.random_state = random_state
        
        # Crear directorios para los resultados si no existen
        self.output_dir = Path("output")
        self.models_dir = self.output_dir / "models"
        self.figures_dir = self.output_dir / "figures"
        self.reports_dir = self.output_dir / "reports"
        
        # Crear directorios si no existen
        for directory in [self.output_dir, self.models_dir, self.figures_dir, self.reports_dir]:
            if not directory.exists():
                directory.mkdir(parents=True)
                
        # Configurar ruta de datos
        if data_path:
            self.data_path = Path(data_path)
        else:
            self.data_path = Path("datasets/housing")
            
        # Atributos a inicializar más tarde
        self.housing = None
        self.strat_train_set = None
        self.strat_test_set = None
        self.housing_prepared = None
        self.housing_labels = None
        self.models = {}
        self.best_model = None
        self.full_pipeline = None
        
        logger.info("Inicialización completada.")
        
    def load_data(self):
        """Carga los datos desde un archivo CSV."""
        try:
            csv_path = self.data_path / "housing.csv"
            logger.info(f"Cargando datos desde {csv_path}")
            self.housing = pd.read_csv(csv_path)
            self.housing = self.housing[self.housing['median_house_value']<=500000].reset_index(drop=True)
            logger.info(f"Datos cargados exitosamente. Shape: {self.housing.shape}")
            return self.housing
        except Exception as e:
            logger.error(f"Error al cargar datos: {e}")
            raise
            
    def explore_data(self):
        """Realiza un análisis exploratorio básico de los datos."""
        if self.housing is None:
            self.load_data()
            
        logger.info("Iniciando exploración de datos...")
        
        
        # Información básica
        info = {
            "shape": self.housing.shape,
            "columns": list(self.housing.columns),
            "dtypes": self.housing.dtypes.to_dict(),
            "missing_values": self.housing.isnull().sum().to_dict(),
            "description": self.housing.describe().to_dict()
        }
        
        # Guardar resultados
        with open(self.reports_dir / "data_exploration.txt", "w") as f:
            f.write("EXPLORACIÓN DE DATOS\n")
            f.write(f"Forma del dataset: {info['shape']}\n\n")
            f.write("Columnas:\n")
            for col in info['columns']:
                f.write(f"- {col} ({info['dtypes'][col]})\n")
            f.write("\nValores faltantes:\n")
            for col, missing in info['missing_values'].items():
                if missing > 0:
                    f.write(f"- {col}: {missing} ({missing/self.housing.shape[0]:.2%})\n")
                    
        logger.info("Exploración básica completada.")
        return info
            
    def visualize_data(self):
        """Genera visualizaciones para entender mejor los datos."""
        if self.housing is None:
            self.load_data()
            
        logger.info("Generando visualizaciones...")
        
        # Configurar estilo de las visualizaciones
        plt.style.use('seaborn-v0_8-whitegrid')
        
        # 1. Histogramas de todas las variables numéricas
        self.housing.hist(bins=50, figsize=(20, 15), grid=False)
        plt.suptitle("Distribución de Variables Numéricas", fontsize=16)
        plt.savefig(self.figures_dir / "numeric_distributions.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Mapa geográfico con precios y densidad poblacional
        plt.figure(figsize=(12, 8))
        plt.scatter(self.housing["longitude"], self.housing["latitude"], 
                    alpha=0.4, s=self.housing["population"]/100,
                    c=self.housing["median_house_value"], cmap="viridis")
        plt.colorbar(label="Precio Medio de Vivienda ($)")
        plt.xlabel("Longitud")
        plt.ylabel("Latitud")
        plt.title("Distribución Geográfica de Precios de Vivienda en California", fontsize=14)
        plt.savefig(self.figures_dir / "geographic_prices.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Matriz de correlaciones
        plt.figure(figsize=(12, 10))
        correlation_matrix = self.housing.corr().round(2)
        sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
        plt.title("Matriz de Correlación entre Variables", fontsize=14)
        plt.tight_layout()
        plt.savefig(self.figures_dir / "correlation_matrix.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # 4. Relación entre ingresos y precios de vivienda
        plt.figure(figsize=(10, 6))
        plt.scatter(self.housing["median_income"], self.housing["median_house_value"], alpha=0.6)
        plt.xlabel("Ingreso Medio")
        plt.ylabel("Precio Medio de Vivienda ($)")
        plt.title("Relación entre Ingresos y Precios de Vivienda", fontsize=14)
        z = np.polyfit(self.housing["median_income"], self.housing["median_house_value"], 1)
        p = np.poly1d(z)
        plt.plot(self.housing["median_income"], p(self.housing["median_income"]), "r--", linewidth=2)
        plt.savefig(self.figures_dir / "income_vs_price.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # 5. Distribución de la variable objetivo
        plt.figure(figsize=(10, 6))
        sns.histplot(self.housing["median_house_value"], kde=True)
        plt.title("Distribución de Precios de Vivienda", fontsize=14)
        plt.xlabel("Precio Medio de Vivienda ($)")
        plt.savefig(self.figures_dir / "target_distribution.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # 6. Box plots para "ocean_proximity"
        plt.figure(figsize=(12, 6))
        sns.boxplot(x="ocean_proximity", y="median_house_value", data=self.housing)
        plt.title("Precios de Vivienda por Proximidad al Océano", fontsize=14)
        plt.xlabel("Proximidad al Océano")
        plt.ylabel("Precio Medio de Vivienda ($)")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(self.figures_dir / "ocean_proximity_boxplot.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info("Visualizaciones generadas y guardadas en el directorio de figuras.")
            
    def split_data(self):
        """Divide los datos en conjuntos de entrenamiento y prueba de manera estratificada."""
        if self.housing is None:
            self.load_data()
            
        logger.info("Dividiendo datos en conjuntos de entrenamiento y prueba...")
        
        # Crear categorías de ingreso para estratificación
        self.housing["income_cat"] = pd.cut(
            self.housing["median_income"],
            bins=[0., 1.5, 3.0, 4.5, 6., np.inf],
            labels=[1, 2, 3, 4, 5]
        )
        
        # División estratificada
        split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=self.random_state)
        for train_index, test_index in split.split(self.housing, self.housing["income_cat"]):
            self.strat_train_set = self.housing.loc[train_index]
            self.strat_test_set = self.housing.loc[test_index]
        
        # Eliminar la categoría de ingresos
        for set_ in (self.strat_train_set, self.strat_test_set):
            set_.drop("income_cat", axis=1, inplace=True)
            
        logger.info(f"Datos divididos. Conjunto de entrenamiento: {self.strat_train_set.shape}, Conjunto de prueba: {self.strat_test_set.shape}")
        
        # Separar características y etiquetas
        self.housing = self.strat_train_set.drop("median_house_value", axis=1)
        self.housing_labels = self.strat_train_set["median_house_value"].copy()
        
    class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
        """Transformer personalizado para añadir características combinadas."""
        
        def __init__(self, add_bedrooms_per_room=True):
            self.add_bedrooms_per_room = add_bedrooms_per_room
            
        def fit(self, X, y=None):
            return self
            
        def transform(self, X):
            X_array = X.values if hasattr(X, 'values') else X
            
            # Obtener índices si X es un DataFrame, de lo contrario usar posiciones fijas
            if hasattr(X, 'columns'):
                rooms_ix = X.columns.get_loc("total_rooms")
                bedrooms_ix = X.columns.get_loc("total_bedrooms")
                population_ix = X.columns.get_loc("population")
                households_ix = X.columns.get_loc("households")
            else:
                # Posiciones fijas basadas en el orden original
                rooms_ix, bedrooms_ix, population_ix, households_ix = 3, 4, 5, 6
            
            rooms_per_household = X_array[:, rooms_ix] / X_array[:, households_ix]
            population_per_household = X_array[:, population_ix] / X_array[:, households_ix]
            
            if self.add_bedrooms_per_room:
                bedrooms_per_room = X_array[:, bedrooms_ix] / X_array[:, rooms_ix]
                return np.c_[X_array, rooms_per_household, population_per_household, bedrooms_per_room]
            else:
                return np.c_[X_array, rooms_per_household, population_per_household]
                
    def preprocess_data(self):
        """Preprocesa los datos para modelado."""
        if self.housing is None or self.housing_labels is None:
            self.split_data()
            
        logger.info("Preprocesando datos...")
        
        # Definir atributos numéricos y categóricos
        num_attribs = list(self.housing.select_dtypes(include=[np.number]))
        cat_attribs = list(self.housing.select_dtypes(exclude=[np.number]))
        
        # Pipeline para atributos numéricos
        num_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy="median")),
            ('attribs_adder', self.CombinedAttributesAdder()),
            ('std_scaler', StandardScaler())
        ])
        
        # Pipeline completo combinando numéricos y categóricos
        self.full_pipeline = ColumnTransformer([
            ("num", num_pipeline, num_attribs),
            ("cat", OneHotEncoder(handle_unknown='ignore'), cat_attribs)
        ])
        
        # Aplicar transformación
        self.housing_prepared = self.full_pipeline.fit_transform(self.housing)
        
        # Obtener nombres de características
        self.feature_names = self._get_feature_names()
        
        logger.info(f"Datos preprocesados. Shape: {self.housing_prepared.shape}")
        logger.info(f"Características generadas: {len(self.feature_names)}")
        
        return self.housing_prepared
        
    def _get_feature_names(self):
        """Obtiene los nombres de las características después del preprocesamiento."""
        # Obtener nombres de características numéricas y sus derivados
        num_attribs = list(self.housing.select_dtypes(include=[np.number]))
        cat_attribs = list(self.housing.select_dtypes(exclude=[np.number]))
        
        # Agregar nombres para características adicionales
        additional_features = ["rooms_per_household", "population_per_household", "bedrooms_per_room"]
        
        # Obtener categorías para características categóricas
        encoder = self.full_pipeline.named_transformers_["cat"]
        cat_features = []
        for i, category in enumerate(cat_attribs):
            # Obtener categorías
            categories = list(encoder.categories_[i])
            # Crear nombres de características
            cat_features.extend([f"{category}_{c}" for c in categories])
        
        # Combinar todas las características
        all_features = num_attribs + additional_features + cat_features
        
        return all_features
        
    def train_and_evaluate_models(self):
        """Entrena y evalúa múltiples modelos."""
        if self.housing_prepared is None:
            self.preprocess_data()
            
        logger.info("Entrenando y evaluando modelos...")
        
        # Definir modelos a entrenar
        self.models = {
            "LinearRegression": LinearRegression(),
            "ElasticNet": ElasticNet(random_state=self.random_state),
            "RandomForest": RandomForestRegressor(random_state=self.random_state),
            "GradientBoosting": GradientBoostingRegressor(random_state=self.random_state)
        }
        
        # Parámetros para búsqueda de hiperparámetros
        param_grids = {
            "LinearRegression": {},
            "ElasticNet": {
                "alpha": [0.1, 0.5, 1.0],
                "l1_ratio": [0.1, 0.5, 0.9]
            },
            "RandomForest": {
                "n_estimators": [50, 100, 200],
                "max_features": [0.3, 0.5, 0.7],
                "max_depth": [10, 20, None]
            },
            "GradientBoosting": {
                "n_estimators": [100, 200],
                "learning_rate": [0.05, 0.1, 0.2],
                "max_depth": [3, 5]
            }
        }
        
        # Resultados y mejor modelo
        best_score = -np.inf
        self.model_results = {}
        
        # Entrenar y evaluar cada modelo
        for name, model in self.models.items():
            logger.info(f"Entrenando modelo: {name}")
            
            # Configurar búsqueda de hiperparámetros
            if param_grids[name]:
                search = GridSearchCV(
                    model, param_grids[name], cv=5,
                    scoring="neg_root_mean_squared_error",
                    return_train_score=True
                )
                search.fit(self.housing_prepared, self.housing_labels)
                self.models[name] = search.best_estimator_
                cv_rmse = -search.best_score_
                best_params = search.best_params_
                logger.info(f"Mejores parámetros para {name}: {best_params}")
            else:
                model.fit(self.housing_prepared, self.housing_labels)
                cv_scores = cross_val_score(
                    model, self.housing_prepared, self.housing_labels,
                    scoring="neg_root_mean_squared_error", cv=5
                )
                cv_rmse = -np.mean(cv_scores)
            
            logger.info(f"Validación cruzada RMSE para {name}: {cv_rmse:.0f}")
            
            # Guardar resultados
            self.model_results[name] = {
                "cv_rmse": cv_rmse,
                "model": self.models[name]
            }
            
            # Actualizar mejor modelo
            if -cv_rmse > best_score:
                best_score = -cv_rmse
                self.best_model = name
        
        logger.info(f"Mejor modelo: {self.best_model} con RMSE: {self.model_results[self.best_model]['cv_rmse']:.0f}")
        
        # Guardar informe de comparación de modelos
        self._save_model_comparison_report()
        
        return self.model_results
    
    def _save_model_comparison_report(self):
        """Genera y guarda un informe comparativo de modelos."""
        with open(self.reports_dir / "model_comparison.txt", "w") as f:
            f.write("COMPARACIÓN DE MODELOS\n")
            f.write("=====================\n\n")
            
            # Ordenar modelos por desempeño
            sorted_models = sorted(
                self.model_results.items(),
                key=lambda x: x[1]["cv_rmse"]
            )
            
            for name, result in sorted_models:
                f.write(f"Modelo: {name}\n")
                f.write(f"RMSE (Validación Cruzada): {result['cv_rmse']:.2f}\n")
                f.write(f"Parámetros: {result['model']}\n\n")
                
            f.write(f"Mejor modelo: {self.best_model}\n")
        
        # Guardar gráfico de comparación
        plt.figure(figsize=(10, 6))
        model_names = list(self.model_results.keys())
        rmse_values = [result["cv_rmse"] for result in self.model_results.values()]
        
        bars = plt.bar(model_names, rmse_values, color='skyblue')
        
        # Resaltar el mejor modelo
        best_index = model_names.index(self.best_model)
        bars[best_index].set_color('green')
        
        plt.ylabel('RMSE (menor es mejor)')
        plt.title('Comparación de Modelos: Error Cuadrático Medio')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(self.figures_dir / "model_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()
        
    def feature_importance_analysis(self):
        """Analiza la importancia de las características para el mejor modelo."""
        if not hasattr(self, "best_model") or self.best_model is None:
            logger.error("Debes entrenar los modelos primero llamando a train_and_evaluate_models().")
            return
            
        model = self.models[self.best_model]
        
        # Verificar si el modelo tiene atributo feature_importances_
        if hasattr(model, "feature_importances_"):
            logger.info("Analizando importancia de características basada en el modelo...")
            importances = model.feature_importances_
            
            # Crear un DataFrame para visualizar
            feature_importance_df = pd.DataFrame({
                'Feature': self.feature_names,
                'Importance': importances
            }).sort_values('Importance', ascending=False)
            
            # Guardar en CSV
            feature_importance_df.to_csv(self.reports_dir / "feature_importance.csv", index=False)
            
            # Visualizar las características más importantes
            plt.figure(figsize=(12, 8))
            plt.barh(
                feature_importance_df['Feature'][:15],
                feature_importance_df['Importance'][:15]
            )
            plt.xlabel('Importancia')
            plt.title(f'Top 15 Características Más Importantes ({self.best_model})')
            plt.gca().invert_yaxis()  # Para que la más importante esté arriba
            plt.tight_layout()
            plt.savefig(self.figures_dir / "feature_importance.png", dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info("Análisis de importancia de características completado.")
            return feature_importance_df
        else:
            logger.info("Calculando importancia de características usando permutación...")
            
            # Usar permutation importance
            result = permutation_importance(
                model, self.housing_prepared, self.housing_labels,
                n_repeats=10, random_state=self.random_state
            )
            
            importances = result.importances_mean
            
            # Crear DataFrame
            feature_importance_df = pd.DataFrame({
                'Feature': self.feature_names,
                'Importance': importances
            }).sort_values('Importance', ascending=False)
            
            # Guardar en CSV
            feature_importance_df.to_csv(self.reports_dir / "feature_importance.csv", index=False)
            
            # Visualizar
            plt.figure(figsize=(12, 8))
            plt.barh(
                feature_importance_df['Feature'][:15],
                feature_importance_df['Importance'][:15]
            )
            plt.xlabel('Importancia (Reducción en rendimiento)')
            plt.title(f'Top 15 Características Más Importantes por Permutación ({self.best_model})')
            plt.gca().invert_yaxis()
            plt.tight_layout()
            plt.savefig(self.figures_dir / "feature_importance_permutation.png", dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info("Análisis de importancia de características por permutación completado.")
            return feature_importance_df
    
    def final_model_evaluation(self):
        """Evalúa el modelo final en el conjunto de prueba."""
        if not hasattr(self, "best_model") or self.best_model is None:
            logger.error("Debes entrenar los modelos primero llamando a train_and_evaluate_models().")
            return
            
        logger.info(f"Evaluando modelo final ({self.best_model}) en conjunto de prueba...")
        
        # Preparar datos de prueba
        X_test = self.strat_test_set.drop("median_house_value", axis=1)
        y_test = self.strat_test_set["median_house_value"].copy()
        X_test_prepared = self.full_pipeline.transform(X_test)
        
        # Obtener predicciones
        final_model = self.models[self.best_model]
        final_predictions = final_model.predict(X_test_prepared)
        
        # Calcular métricas
        mse = mean_squared_error(y_test, final_predictions)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, final_predictions)
        r2 = r2_score(y_test, final_predictions)
        
        # Guardar resultados
        final_scores = {
            "model": self.best_model,
            "rmse": rmse,
            "mae": mae,
            "r2": r2
        }
        
        logger.info(f"Evaluación final - RMSE: {rmse:.2f}, MAE: {mae:.2f}, R²: {r2:.4f}")
        
        # Guardar informe
        with open(self.reports_dir / "final_evaluation.txt", "w") as f:
            f.write("EVALUACIÓN DEL MODELO FINAL\n")
            f.write("=========================\n\n")
            f.write(f"Modelo: {self.best_model}\n")
            f.write(f"RMSE: {rmse:.2f}\n")
            f.write(f"MAE: {mae:.2f}\n")
            f.write(f"R²: {r2:.4f}\n\n")
            f.write("Interpretación:\n")
            f.write(f"- El error cuadrático medio (RMSE) de {rmse:.2f} significa que el modelo tiene un error promedio de ")
            f.write(f"aproximadamente ${rmse:.0f} en las predicciones de precios de vivienda.\n")
            f.write(f"- El coeficiente de determinación (R²) de {r2:.4f} indica que el modelo explica ")
            f.write(f"aproximadamente el {r2*100:.1f}% de la varianza en los precios de vivienda.\n")
        
        # Visualizar predicciones vs valores reales
        plt.figure(figsize=(10, 7))
        plt.scatter(y_test, final_predictions, alpha=0.5)
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
        plt.xlabel('Precios Reales')
        plt.ylabel('Predicciones')
        plt.title('Predicciones vs Valores Reales de Precios de Vivienda')
        plt.savefig(self.figures_dir / "predictions_vs_actual.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # Visualizar residuos
        residuals = final_predictions - y_test
        plt.figure(figsize=(10, 7))
        plt.scatter(final_predictions, residuals, alpha=0.5)
        plt.axhline(y=0, color='r', linestyle='-')
        plt.xlabel('Predicciones')
        plt.ylabel('Residuos')
        plt.title('Residuos vs Predicciones')
        plt.savefig(self.figures_dir / "residuals.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # Histograma de residuos
        plt.figure(figsize=(10, 7))
        sns.histplot(residuals, kde=True)
        plt.axvline(x=0, color='r', linestyle='-')
        plt.xlabel('Residuos')
        plt.title('Distribución de Residuos')
        plt.savefig(self.figures_dir / "residuals_distribution.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        return final_scores
        
    def save_model(self):
        """Guarda el modelo final y el pipeline de preprocesamiento."""
        if not hasattr(self, "best_model") or self.best_model is None:
            logger.error("Debes entrenar los modelos primero llamando a train_and_evaluate_models().")
            return
            
        logger.info(f"Guardando modelo final ({self.best_model})...")
        
        # Crear timestamp para el nombre del archivo
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Guardar modelo
        model_file = self.models_dir / f"housing_model_{self.best_model}_{timestamp}.pkl"
        pipeline_file = self.models_dir / f"preprocessing_pipeline_{timestamp}.pkl"
        
        joblib.dump(self.models[self.best_model], model_file)
        joblib.dump(self.full_pipeline, pipeline_file)
        
        logger.info(f"Modelo guardado en {model_file}")
        logger.info(f"Pipeline guardado en {pipeline_file}")
        
        # Guardar información del modelo
        model_info = {
            "model_type": self.best_model,
            "model_file": str(model_file),
            "pipeline_file": str(pipeline_file),
            "training_date": timestamp,
            "features": self.feature_names,
            "metrics": {
                "rmse": self.model_results[self.best_model]["cv_rmse"]
            }
        }
        
        with open(self.models_dir / f"model_info_{timestamp}.txt", "w") as f:
            f.write("INFORMACIÓN DEL MODELO\n")
            f.write("=====================\n\n")
            f.write(f"Tipo de modelo: {model_info['model_type']}\n")
            f.write(f"Archivo del modelo: {model_info['model_file']}\n")
            f.write(f"Archivo del pipeline: {model_info['pipeline_file']}\n")
            f.write(f"Fecha de entrenamiento: {timestamp}\n")
            f.write(f"RMSE (Validación cruzada): {model_info['metrics']['rmse']:.2f}\n\n")
            f.write("Características utilizadas:\n")
            for feature in model_info['features']:
                f.write(f"- {feature}\n")
                
        logger.info("Información del modelo guardada.")
        return model_info
        
    def train_pipeline(self):
        """Ejecuta el flujo completo de entrenamiento de modelos, desde la carga de datos hasta la evaluación final."""
        logger.info("Iniciando pipeline completo de entrenamiento...")

        # Paso 1: Cargar datos
        self.load_data()

        # Paso 2: Explorar y visualizar datos
        self.explore_data()
        self.visualize_data()

        # Paso 3: Dividir y preprocesar datos
        self.split_data()
        self.preprocess_data()

        # Paso 4: Entrenar y evaluar modelos
        self.train_and_evaluate_models()

        # Paso 5: Analizar importancia de características
        self.feature_importance_analysis()

        # Paso 6: Evaluación final del mejor modelo
        final_results = self.final_model_evaluation()

        # Paso 7: Guardar el modelo final
        model_info = self.save_model()

        logger.info("Pipeline de entrenamiento completado exitosamente.")
        logger.info(f"Mejor modelo: {self.best_model} con RMSE: {final_results['rmse']:.2f}")

        return {
            "best_model": self.best_model,
            "metrics": final_results,
            "model_info": model_info
        }

    def predict(self, housing_data):
        """
        Realiza predicciones utilizando el modelo entrenado.

        Parameters:
        -----------
        housing_data : pandas.DataFrame
            Datos de vivienda para realizar predicciones (sin la columna median_house_value)

        Returns:
        --------
        numpy.ndarray
            Predicciones de precios de vivienda
        """
        if not hasattr(self, "best_model") or self.best_model is None:
            logger.error("No hay un modelo entrenado disponible para predicciones.")
            raise ValueError("Entrena un modelo primero usando train_pipeline() o train_and_evaluate_models().")

        logger.info("Realizando predicciones con el modelo entrenado...")

        # Preprocesar datos de entrada
        X_prepared = self.full_pipeline.transform(housing_data)

        # Realizar predicciones
        predictions = self.models[self.best_model].predict(X_prepared)

        logger.info(f"Predicciones completadas para {len(predictions)} instancias.")

        return predictions
    
    def load_model(self, model_path, pipeline_path):
        """
        Carga un modelo previamente entrenado y su pipeline de preprocesamiento.

        Parameters:
        -----------
        model_path : str or Path
            Ruta al archivo del modelo guardado
        pipeline_path : str or Path
            Ruta al archivo del pipeline de preprocesamiento

        Returns:
        --------
        bool
            True si la carga fue exitosa
        """
        try:
            logger.info(f"Cargando modelo desde {model_path}...")
            model = joblib.load(model_path)

            logger.info(f"Cargando pipeline desde {pipeline_path}...")
            self.full_pipeline = joblib.load(pipeline_path)

            # Determinar tipo de modelo
            if isinstance(model, LinearRegression):
                model_type = "LinearRegression"
            elif isinstance(model, ElasticNet):
                model_type = "ElasticNet"
            elif isinstance(model, RandomForestRegressor):
                model_type = "RandomForest"
            elif isinstance(model, GradientBoostingRegressor):
                model_type = "GradientBoosting"
            else:
                model_type = model.__class__.__name__

            # Configurar modelo cargado
            self.models = {model_type: model}
            self.best_model = model_type

            # Obtener nombres de características
            self.feature_names = self._get_feature_names()

            logger.info(f"Modelo cargado exitosamente. Tipo: {model_type}")
            return True

        except Exception as e:
            logger.error(f"Error al cargar el modelo: {e}")
            return False
    
if __name__ == "__main__":
    # Ejemplo de uso del modelo
    model = HousingPriceModel()
    
    # Entrenar pipeline completo
    results = model.train_pipeline()
    
    print(f"\nEntrenamiento completado.")
    print(f"Mejor modelo: {results['best_model']}")
    print(f"RMSE: {results['metrics']['rmse']:.2f}")
    print(f"R²: {results['metrics']['r2']:.4f}")
    print(f"\nModelo guardado en: {results['model_info']['model_file']}")
    
    # Ejemplo de cómo cargar un modelo guardado
    # saved_model = HousingPriceModel()
    # saved_model.load_model(
    #     results['model_info']['model_file'],
    #     results['model_info']['pipeline_file']
    # )
    
    # Ejemplo de cómo hacer predicciones
    # new_data = pd.read_csv("nuevos_datos.csv")
    # predictions = saved_model.predict(new_data)