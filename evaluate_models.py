def evaluate_model(y_observed, y_predicted, loss_function="mse"):
    if loss_function == "mse":
        return sum((y_observed - y_predicted)**2) / len(y_observed)
    elif loss_function == "rmse":
        return (sum((y_observed - y_predicted) ** 2) / len(y_observed))**0.5
