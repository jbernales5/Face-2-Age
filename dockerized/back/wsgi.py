from server import app, load_model_age_real, load_model_gender

if __name__ == "__main__":
    load_model_age_real()
    load_model_gender()
    app.run()
