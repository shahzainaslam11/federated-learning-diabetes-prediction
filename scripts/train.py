import yaml
from src.data.preprocessing import DiabetesPreprocessor
from src.models.factory import get_model
from src.training.cross_validation import CrossValidator
from src.utils.seed import set_seed

def main():
    config = yaml.safe_load(open("configs/config.yaml"))
    set_seed(config["seed"])

    preprocessor = DiabetesPreprocessor()
    X, y = preprocessor.preprocess(config["data"]["path"])

    model = get_model("xgb", config["models"]["xgb"])

    cv = CrossValidator(model, config["training"])
    results = cv.run(X, y)

    print(results)

if __name__ == "__main__":
    main()
