import os
import pandas as pd
import mlflow
from datetime import datetime

def register_dataset_as_provenance(manifest_path, dataset_root, dataset_name, version):
    """
    Zaregistruje dataset do MLflow.
    Používá parametry pro indexaci a artefakty jako zdroj pravdy (PROV-O připraveno).
    """
    mlflow.set_experiment("Dataset_Registry")
    
    with mlflow.start_run(run_name=f"Dataset_{dataset_name}_v{version}") as run:
        # 1. Indexace v MLflow (pro rychlé hledání/filtrování)
        mlflow.set_tag("dataset_name", dataset_name)
        mlflow.set_tag("version", version)
        
        # 2. Metadata pro tracking
        mlflow.log_param("dataset_root", dataset_root)
        
        # 3. Zpracování manifestu a obohacení o metadata (size, mtime)
        df = pd.read_csv(manifest_path)
        metadata_list = []
        
        for path in df['wsi_path']:
            full_path = os.path.join(dataset_root, path) if not os.path.isabs(path) else path
            if os.path.exists(full_path):
                stat = os.stat(full_path)
                metadata_list.append({"file_size": stat.st_size, "last_modified": stat.st_mtime})
            else:
                metadata_list.append({"file_size": -1, "last_modified": -1})
        
        df_enriched = pd.concat([df, pd.DataFrame(metadata_list)], axis=1)
        
        # 4. Uložení artefaktu (Zlatý zdroj pravdy)
        # Toto CSV budeš později skenovat pro tvorbu JSON-LD (PROV-O)
        provenance_file = "dataset_provenance.csv"
        df_enriched.to_csv(provenance_file, index=False)
        mlflow.log_artifact(provenance_file, artifact_path="provenance")
        
        # 5. Uložení odkazu do tagu (velmi důležité pro automatizaci!)
        # Uložíme si, kde v artefaktech to CSV leží
        mlflow.set_tag("manifest_uri", f"runs:/{run.info.run_id}/provenance/{provenance_file}")
        
        print(f"Dataset '{dataset_name}' (v{version}) úspěšně zaregistrován.")
        print(f"Run ID: {run.info.run_id}")
        
        os.remove(provenance_file)

if __name__ == "__main__":
    # Příklad použití pro tvůj dataset
    register_dataset_as_provenance(
        manifest_path="data/dummy_dataset_1/manifest.csv",
        dataset_root="data/dummy_dataset_1",
        dataset_name="pato_cohort_01",
        version="1.0.0"
    )