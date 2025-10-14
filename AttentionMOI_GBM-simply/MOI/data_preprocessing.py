import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from scipy import sparse
import os
import warnings
import joblib

warnings.filterwarnings('ignore')


class MultiOmicsDataPreprocessor:
    """Preprocessor for multi-omics TCGA data_RNA integration and preprocessing."""

    def __init__(self, data_dir, output_dir):
        self.data_dir = data_dir
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        self.label_encoder = LabelEncoder()
        self.dna_scaler = StandardScaler()
        self.rna_scaler = StandardScaler()
        self.full_re = re.compile(r'^(TCGA-[A-Z0-9]{2}-[A-Z0-9]{4})-([0-9]{2})[A-Z]?.*')

    def _extract_patient_id(self, barcode):
        """Extract patient ID from TCGA barcode."""
        if not isinstance(barcode, str):
            return None
        match = self.full_re.match(barcode.strip())
        return match.group(1) if match else None

    def _extract_sample_code(self, barcode):
        """Extract sample type code from TCGA barcode."""
        if not isinstance(barcode, str):
            return None
        match = self.full_re.match(barcode.strip())
        return match.group(2) if match else None

    def _is_primary_tumor(self, barcode):
        """Check if sample is primary tumor (sample code 01)."""
        code = self._extract_sample_code(barcode)
        return True if code is None else (code == "01")

    def _read_tsv(self, filepath):
        """Read TSV file with fallback encoding."""
        try:
            return pd.read_csv(filepath, sep='\t', low_memory=False)
        except Exception:
            return pd.read_csv(filepath, sep='\t', encoding='latin-1', low_memory=False)

    def load_clinical_data(self, clinical_file):
        """Load and preprocess clinical data_RNA with subtype information."""
        df = self._read_tsv(clinical_file)

        # Detect data_RNA format: matrix (attributes x samples) or sample-based
        tcga_cols = [c for c in df.columns if str(c).startswith("TCGA-")]

        if len(tcga_cols) >= 5:
            # Matrix format: find subtype row
            attr_col = df.columns[0]
            subtype_mask = df[attr_col].astype(str).str.contains(
                r'(subtype|GeneExp_Subtype|Subtype_mRNA|mRNA.*cluster|verhaak)',
                case=False, regex=True, na=False
            )
            if not subtype_mask.any():
                raise ValueError("Subtype row not found in clinical matrix")
            subtype_row = df.loc[subtype_mask].iloc[0]
            clin = pd.DataFrame({
                "sample_barcode": subtype_row.index[1:],
                "subtype": subtype_row.values[1:]
            })
        else:
            # Sample-based format
            sid_col = next((c for c in ["sampleID", "bcr_sample_barcode", "sample", "barcode"]
                            if c in df.columns), df.columns[0])
            subtype_col = next((c for c in ["GeneExp_Subtype", "Subtype_mRNA", "Subtype", "subtype"]
                                if c in df.columns), None)
            if not subtype_col:
                cands = [c for c in df.columns if 'subtype' in str(c).lower()]
                subtype_col = cands[0] if cands else None
            if not subtype_col:
                raise ValueError("Subtype column not found in clinical data_RNA")
            clin = df[[sid_col, subtype_col]].copy()
            clin.columns = ["sample_barcode", "subtype"]

        # Clean and standardize data_RNA
        clin["sample_barcode"] = clin["sample_barcode"].astype(str)
        clin["patient_id"] = clin["sample_barcode"].apply(self._extract_patient_id)
        clin["is_primary"] = clin["sample_barcode"].apply(self._is_primary_tumor)
        clin["subtype"] = clin["subtype"].apply(
            lambda x: str(x)[:1].upper() + str(x)[1:].lower() if pd.notna(x) else x
        )

        # Filter and deduplicate
        clin = clin[clin["patient_id"].notna() & clin["subtype"].notna()]
        clin = (clin.sort_values(["patient_id", "is_primary"], ascending=[True, False])
                .drop_duplicates("patient_id", keep="first"))

        result = clin[["patient_id", "subtype"]].rename(columns={"patient_id": "short_id"})
        print(f"Clinical data_RNA (deduplicated): {len(result)} patients")
        return result

    def load_omics_data(self, omics_file, data_type="methylation"):
        """Load and transpose omics data_RNA matrix."""
        df = self._read_tsv(omics_file)

        # Set first column as index if it contains feature identifiers
        first_col = str(df.columns[0]).strip().lower()
        feature_keys = {"gene", "genes", "hugo", "symbol", "gene symbol", "id", "probe"}
        if first_col in feature_keys:
            df = df.set_index(df.columns[0])

        # Transpose if samples are in rows
        if not any(str(c).startswith("TCGA-") for c in df.columns):
            if any(str(i).startswith("TCGA-") for i in df.index):
                df = df.transpose()

        # Select TCGA samples and convert to numeric
        tcga_cols = [c for c in df.columns if str(c).startswith("TCGA-")]
        if not tcga_cols:
            raise ValueError(f"No TCGA samples found in {data_type} data_RNA")

        df = df[tcga_cols].apply(pd.to_numeric, errors="coerce")
        return df.transpose()

    def match_samples(self, omics_data, clinical_data):
        """Match omics samples with clinical data_RNA and extract labels."""
        patient_ids = omics_data.index.to_series().astype(str).apply(self._extract_patient_id)
        valid_mask = patient_ids.notna()
        omics_data = omics_data.loc[valid_mask]
        patient_ids = patient_ids.loc[valid_mask]

        # Remove duplicate patients
        unique_mask = ~patient_ids.duplicated()
        omics_data = omics_data.loc[unique_mask]
        patient_ids = patient_ids.loc[unique_mask]

        # Map to subtypes
        patient_to_subtype = dict(zip(clinical_data["short_id"], clinical_data["subtype"]))
        labels = patient_ids.map(patient_to_subtype)
        matched_mask = labels.notna()

        X_matched = omics_data.loc[matched_mask]
        y_matched = labels.loc[matched_mask]
        X_matched.index = patient_ids.loc[matched_mask].values
        y_matched.index = X_matched.index

        return X_matched, y_matched

    def process_large_matrix(self, matrix, batch_size=1000, fill_method='mean'):
        """Process large matrix in batches to handle missing values."""
        n_features = matrix.shape[1]
        n_batches = (n_features + batch_size - 1) // batch_size
        batches = []

        for i in range(n_batches):
            start = i * batch_size
            end = min((i + 1) * batch_size, n_features)
            batch = matrix.iloc[:, start:end].copy()

            if fill_method == 'mean':
                batch = batch.fillna(batch.mean())
            elif fill_method == 'median':
                batch = batch.fillna(batch.median())
            elif fill_method == 'zero':
                batch = batch.fillna(0)
            else:
                batch = batch.fillna(batch.mean())

            batches.append(batch)

        return pd.concat(batches, axis=1)

    def preprocess_data(self, clinical_file, mutation_as_main_file, cnv_file, expression_file, mutation_file=None):
        """Main preprocessing pipeline for multi-omics data_RNA integration."""
        print("Loading clinical data_RNA...")
        clinical_data = self.load_clinical_data(clinical_file)
        if len(clinical_data) == 0:
            raise ValueError("Clinical data_RNA is empty")

        print("Loading mutation data_RNA (main modality)...")
        mut_data = self.load_omics_data(mutation_as_main_file, "mutation")
        print(f"Mutation data_RNA: {mut_data.shape}")

        print("Loading CNV data_RNA...")
        cnv_data = self.load_omics_data(cnv_file, "CNV")
        print(f"CNV data_RNA: {cnv_data.shape}")

        print("Loading expression data_RNA...")
        expr_data = self.load_omics_data(expression_file, "expression")
        print(f"Expression data_RNA: {expr_data.shape}")

        print("Matching samples...")
        mut_matched, y_mut = self.match_samples(mut_data, clinical_data)
        cnv_matched, y_cnv = self.match_samples(cnv_data, clinical_data)
        expr_matched, y_expr = self.match_samples(expr_data, clinical_data)

        # Find common samples across all modalities - FIX: Convert set to sorted list
        common_samples = sorted(list(set(mut_matched.index) & set(cnv_matched.index) & set(expr_matched.index)))
        print(f"Common samples: {len(common_samples)}")
        if len(common_samples) == 0:
            raise ValueError("No common samples found")

        print("Processing matrices...")
        mut_processed = self.process_large_matrix(mut_matched.loc[common_samples])
        cnv_processed = self.process_large_matrix(cnv_matched.loc[common_samples])
        expr_processed = self.process_large_matrix(expr_matched.loc[common_samples])
        y_final = y_mut.loc[common_samples]

        # Combine DNA modalities
        dna_data = pd.concat([cnv_processed, mut_processed], axis=1)
        rna_data = expr_processed

        print(f"Final data_RNA shapes - DNA: {dna_data.shape}, RNA: {rna_data.shape}")

        # Split data_RNA
        y_encoded = self.label_encoder.fit_transform(y_final.values)
        X_dna_train, X_dna_test, X_rna_train, X_rna_test, y_train, y_test = train_test_split(
            dna_data.values, rna_data.values, y_encoded,
            test_size=0.2, random_state=42, stratify=y_encoded
        )
        X_dna_train, X_dna_val, X_rna_train, X_rna_val, y_train, y_val = train_test_split(
            X_dna_train, X_rna_train, y_train,
            test_size=0.2, random_state=42, stratify=y_train
        )

        # Standardize features
        self.dna_scaler.fit(X_dna_train)
        self.rna_scaler.fit(X_rna_train)

        X_dna_train = self.dna_scaler.transform(X_dna_train)
        X_dna_val = self.dna_scaler.transform(X_dna_val)
        X_dna_test = self.dna_scaler.transform(X_dna_test)
        X_rna_train = self.rna_scaler.transform(X_rna_train)
        X_rna_val = self.rna_scaler.transform(X_rna_val)
        X_rna_test = self.rna_scaler.transform(X_rna_test)

        # Save processed data_RNA
        sparse.save_npz(os.path.join(self.output_dir, 'dna_train.npz'), sparse.csr_matrix(X_dna_train))
        sparse.save_npz(os.path.join(self.output_dir, 'dna_val.npz'), sparse.csr_matrix(X_dna_val))
        sparse.save_npz(os.path.join(self.output_dir, 'dna_test.npz'), sparse.csr_matrix(X_dna_test))
        sparse.save_npz(os.path.join(self.output_dir, 'rna_train.npz'), sparse.csr_matrix(X_rna_train))
        sparse.save_npz(os.path.join(self.output_dir, 'rna_val.npz'), sparse.csr_matrix(X_rna_val))
        sparse.save_npz(os.path.join(self.output_dir, 'rna_test.npz'), sparse.csr_matrix(X_rna_test))

        np.save(os.path.join(self.output_dir, 'y_train.npy'), y_train)
        np.save(os.path.join(self.output_dir, 'y_val.npy'), y_val)
        np.save(os.path.join(self.output_dir, 'y_test.npy'), y_test)

        joblib.dump(self.label_encoder, os.path.join(self.output_dir, 'label_encoder.pkl'))
        joblib.dump(self.dna_scaler, os.path.join(self.output_dir, 'dna_scaler.pkl'))
        joblib.dump(self.rna_scaler, os.path.join(self.output_dir, 'rna_scaler.pkl'))

        print(f"Preprocessing complete. Data saved to: {self.output_dir}")
        return {
            'dna_train': X_dna_train, 'dna_val': X_dna_val, 'dna_test': X_dna_test,
            'rna_train': X_rna_train, 'rna_val': X_rna_val, 'rna_test': X_rna_test,
            'y_train': y_train, 'y_val': y_val, 'y_test': y_test
        }


def main():
    """Example usage of multi-omics data_RNA preprocessor."""
    output_dir = "./preprocessed"
    data_files = {
        'clinical': './data_RNA/TCGA.GBM.sampleMap_GBM_clinicalMatrix',
        'methylation': './data_RNA/HumanMethylation450',
        'cnv': './data_RNA/Gistic2_CopyNumber_Gistic2_all_thresholded.by_genes',
        'expression': './data_RNA/HiSeqV2',
        'mutation': './data_RNA/GBM_mc3_gene_level.txt'
    }

    """CNV + mutations merged as DNA,
     expression HiSeqV2 as RNA modality"""
    preprocessor = MultiOmicsDataPreprocessor(data_dir="./data_RNA", output_dir=output_dir)
    preprocessor.preprocess_data(
        clinical_file=data_files['clinical'],
        mutation_as_main_file=data_files['mutation'],
        cnv_file=data_files['cnv'],
        expression_file=data_files['expression']
    )


if __name__ == "__main__":
    main()