import mlflow

def register_new_user(username, real_name, email, organization, lead_name, lead_email):
    """
    Registruje uživatele do MLflow experimentu 'User_Registry' s kompletními údaji.
    """
    experiment_name = "User_Registry"
    experiment = mlflow.get_experiment_by_name(experiment_name)
    
    if not experiment:
        experiment_id = mlflow.create_experiment(experiment_name)
        print(f"Vytvořen experiment: {experiment_name}")
    else:
        experiment_id = experiment.experiment_id

    # Vytvoření unikátního runu pro tohoto uživatele
    with mlflow.start_run(experiment_id=experiment_id, run_name=f"User_{username}"):
        mlflow.set_tags({
            "username": username,
            "real_name": real_name,
            "email": email,
            "organization": organization,
            "lead_name": lead_name,
            "lead_email": lead_email
        })
        print(f"Uživatel {real_name} ({username}) byl úspěšně zaregistrován.")

if __name__ == "__main__":
    # Příklad registrace
    register_new_user(
        username="jiribuchta",
        real_name="Jiří Buchta",
        email="524981@mail.muni.cz",
        organization="RationAI",
        lead_name="Tomáš Brázdil",
        lead_email="brazdil@muni.cz"
    )