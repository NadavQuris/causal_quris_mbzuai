from typing import List, Union

import pandas as pd
import pubchempy
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from tqdm import tqdm


class Molecule:
    @classmethod
    def from_inchi(cls, inchi: str, name: Union[str, List[str]] = None):
        return cls(inchi, name)

    @classmethod
    def from_smiles(cls, smiles: str, name: Union[str, List[str]] = None):
        inchi = canonical_smiles_to_inchi(canonicalize_smiles(smiles))
        return cls(inchi, name)

    def __init__(self, inchi: str, name: Union[str, List[str]] = None):
        self.name = name
        if isinstance(name, str):
            self.name = [name]
        self._inchi = inchi
        self._smiles = inchi_to_canonical_smiles(inchi)
        self._rdkit_mol = None
        self._scaffold = None

    def __eq__(self, other):
        return self.inchi == other.inchi

    @property
    def inchi(self):
        return self._inchi

    @property
    def smiles(self):
        return self._smiles

    @property
    def inchi_key(self):
        return Chem.InchiToInchiKey(self.inchi)

    @property
    def molecule(self):
        if self._rdkit_mol is None:
            self._rdkit_mol = Chem.MolFromInchi(self._inchi)
        return self._rdkit_mol

    @property
    def scaffold(self):
        if self._scaffold is None:
            scaffold = MurckoScaffold.GetScaffoldForMol(self.molecule)
            scaffold_inchi = Chem.MolToInchi(scaffold)
            self._scaffold = Molecule.from_inchi(scaffold_inchi)
        return self._scaffold


def canonicalize_smiles(smiles: str):
    try:
        mol = Chem.MolFromSmiles(smiles)
        return Chem.MolToSmiles(mol)
    except Exception as e:
        return None


def inchi_to_canonical_smiles(inchi: str):
    try:
        mol = Chem.MolFromInchi(inchi)
        return Chem.MolToSmiles(mol, isomericSmiles=True)
    except Exception as e:
        return None  # None means that a SMILES could not be generated from the input


def canonical_smiles_to_inchi(smiles: str):
    try:
        # RDKit
        mol = Chem.MolFromSmiles(smiles)
        return Chem.MolToInchi(mol)
    except Exception as e:
        return None  # None means that an InChI could not be generated from the input


def check_if_smiles_strings_are_equivalent(smiles_1: str, smiles_2: str):
    c_smiles1, c_smiles2 = canonicalize_smiles(smiles_1), canonicalize_smiles(smiles_2)
    return c_smiles1 == c_smiles2


def get_synonyms_from_pubchem(inchi: str, limit: int = None):
    synonyms = pubchempy.get_synonyms(inchi, "inchi")
    if not synonyms:
        return []
    else:
        synonyms = synonyms[0]["Synonym"]
    if limit:
        synonyms = synonyms[:limit]
    return synonyms


def get_molecule_from_pubchem(molecule_name: str, as_dataframe: bool = True):
    mol = pubchempy.get_compounds(molecule_name, "name", as_dataframe=as_dataframe)
    return mol


def get_molecule_from_chembl(molecule_name: str) -> List:
    from chembl_webresource_client.new_client import new_client

    molecule = new_client.molecule
    return molecule.filter(pref_name__iexact=molecule_name)


def add_column_to_df(df: pd.DataFrame, column_name: str, func, force: bool = False):
    if column_name not in df.columns or force:
        assert "inchi" in df.columns
        col_values = []
        for _, row in tqdm(
            df.iterrows(), total=len(df), desc=f"adding column {column_name}"
        ):
            col_values.append(func(row["inchi"]))
        df[column_name] = col_values
    return df


def add_smiles_to_df(df: pd.DataFrame, force: bool = False) -> pd.DataFrame:
    if "smiles" not in df.columns or force:
        assert "inchi" in df.columns
        smiles = []
        for _, row in tqdm(
            df.iterrows(), total=len(df), desc="converting inchi to smiles"
        ):
            try:
                smiles.append(Molecule.from_inchi(row["inchi"]).smiles)
            except:
                smiles.append(None)
        df["smiles"] = smiles
        if (df["smiles"] == None).any():
            # remove and print how many
            print(
                f"dropping {len(df[df['smiles'] == None])} rows, for which smiles could not be generated"
            )
            df = df[df["smiles"] != None]
    return df


def add_scaffold_column_to_df(df, force: bool = False):
    def get_scaffold_from_inchi(inchi: str) -> str:
        try:
            mol = Molecule.from_inchi(inchi)
            scaffold = mol.scaffold.inchi
            return scaffold
        except:
            return "Could not find scaffold"

    return add_column_to_df(df, "scaffold", get_scaffold_from_inchi, force)


def add_chembl_structural_alerts_column_to_df(df, force: bool = False):
    structural_alrets_df = pd.read_csv("resources/chembl_structural_alerts.csv")
    alerts_dict = {}
    for _, row in structural_alrets_df.iterrows():
        alerts_mol = Chem.MolFromSmarts(row["smarts"])
        if row["alert_name"] in alerts_dict:
            alerts_dict[row["alert_name"]].append(alerts_mol)
        else:
            alerts_dict[row["alert_name"]] = [alerts_mol]

    def get_structural_alerts_from_inchi(inchi: str) -> List:
        mol = Molecule.from_inchi(inchi)
        alerts = []
        for alert_name, alert_mol_list in alerts_dict.items():
            for alert_mol in alert_mol_list:
                if (
                    mol.molecule.HasSubstructMatch(alert_mol)
                    and alert_name not in alerts
                ):
                    alerts.append(alert_name)
        return alerts

    return add_column_to_df(
        df, "chembl_structural_alerts", get_structural_alerts_from_inchi, force
    )
