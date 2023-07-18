import re
from typing import List
from urllib.parse import quote
from urllib.request import urlopen

import pandas as pd
from bs4 import BeautifulSoup
from tqdm import tqdm

from mol_utils.utils import (
    inchi_to_canonical_smiles,
    get_molecule_from_pubchem,
    get_molecule_from_chembl,
)

CHEMID_PLUS, PUBCHEM, CHEMBL, CACTUS, WIKIPEDIA = (
    "chemidplus",
    "pubchem",
    "chembl",
    "cactus",
    "wikipedia",
)

PRIORITIZED_SOURCES = [CHEMID_PLUS, PUBCHEM, CHEMBL, WIKIPEDIA]


# ----- getting inchis and smiles from sources ----- #   TODO: make sources into classes? could be useful


class InChINotFoundException(Exception):
    pass


class SMILESNotFoundException(Exception):
    pass


def raises_inchi_not_found(func):
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            raise InChINotFoundException(e)

    return wrapper


def raises_smiles_not_found(func):
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            raise SMILESNotFoundException(e)

    return wrapper


@raises_smiles_not_found
def get_smiles_from_pubchem(molecule_name: str) -> List:
    mol = get_molecule_from_pubchem(molecule_name)
    return list(mol["isomeric_smiles"])


@raises_smiles_not_found
def get_smiles_from_chembl(molecule_name: str) -> List:
    mol_list = get_molecule_from_chembl(molecule_name)
    return [mol["molecule_structures"]["canonical_smiles"] for mol in mol_list]


@raises_smiles_not_found
def get_smiles_from_cactus(molecule_name: str) -> List:
    url = (
        "http://cactus.nci.nih.gov/chemical/structure/"
        + quote(molecule_name)
        + "/smiles"
    )
    ans = urlopen(url).read().decode("utf8")
    return [ans]


@raises_smiles_not_found
def get_smiles_from_wikipedia(molecule_name: str) -> List:
    molecule_name = molecule_name.replace(" ", "_")
    html = urlopen("https://en.wikipedia.org/wiki/" + molecule_name)
    wiki_extract = BeautifulSoup(html, "lxml").get_text()
    smiles_match = re.findall("SMILES\n.*", wiki_extract)
    smiles_clean = smiles_match[0][7:]
    return [smiles_clean]


@raises_smiles_not_found
def get_smiles_from_chemidplus(molecule_name: str) -> List:
    # This is an incomplete way to get smiles from chemidplus.
    # If there are multiple smiles, it will return None
    molecule_name = molecule_name.replace(" ", "_")
    html = urlopen("https://chem.nlm.nih.gov/chemidplus/name/" + molecule_name)
    chemidplus_extract = BeautifulSoup(html, "lxml").get_text()
    smiles_match = re.findall("Smiles\n.*", chemidplus_extract)
    smiles_clean = smiles_match[0][7:]
    return [smiles_clean]


@raises_inchi_not_found
def get_inchi_from_puchem(molecule_name: str) -> List:
    return list(get_molecule_from_pubchem(molecule_name)["inchi"])


@raises_inchi_not_found
def get_inchi_from_chembl(molecule_name: str) -> List:
    mol_list = get_molecule_from_chembl(molecule_name)
    return [mol["molecule_structures"]["standard_inchi"] for mol in mol_list]


@raises_inchi_not_found
def get_inchi_from_cactus(molecule_name: str) -> List:
    url = (
        "http://cactus.nci.nih.gov/chemical/structure/"
        + quote(molecule_name)
        + "/stdinchi"
    )
    ans = urlopen(url).read().decode("utf8")
    return [ans]


@raises_inchi_not_found
def get_inchi_from_wikipedia(molecule_name: str):
    molecule_name = molecule_name.replace(" ", "_")
    html = urlopen("https://en.wikipedia.org/wiki/" + molecule_name)
    wiki_extract = BeautifulSoup(html, "lxml").get_text()
    inchi_matches = re.findall("InChI=.*", wiki_extract)[0]
    if not inchi_matches:
        raise InChINotFoundException(f"No InChI found for {molecule_name} in Wikipedia")
    if isinstance(inchi_matches, list):
        inchi_matches = inchi_matches[0]
    inchi_without_key = inchi_matches.split("Key:")[0]
    inchi_clean = inchi_without_key.split("H\\")
    inchi_final = inchi_clean[0].split()[0]
    return [inchi_final]


@raises_inchi_not_found
def get_inchi_from_chemidplus(molecule_name: str) -> List:
    # This is an incomplete way to get inchis from chemisplus.
    # If there are multiple inchis, it will return None
    molecule_name = molecule_name.replace(" ", "_")
    html = urlopen("https://chem.nlm.nih.gov/chemidplus/name/" + molecule_name)
    chemidplus_extract = BeautifulSoup(html, "lxml").get_text()
    inchi_matches = re.findall("InChI=.*", chemidplus_extract)
    if not inchi_matches:
        raise InChINotFoundException(
            f"No InChI found for {molecule_name} in ChemIDPlus"
        )
    inchi_without_key = inchi_matches[0].split("Key:")[0]
    inchi_clean = inchi_without_key.split("H\\")
    inchi_final = inchi_clean[0].split()[0]
    return [inchi_final]


inchi_functions_dict = {
    CHEMID_PLUS: get_inchi_from_chemidplus,
    PUBCHEM: get_inchi_from_puchem,
    CHEMBL: get_inchi_from_chembl,
    CACTUS: get_inchi_from_cactus,
    WIKIPEDIA: get_inchi_from_wikipedia,
}


# ----- Extraction functions ----- #


def get_best_choice_of_inchi(
    inchi_dict: dict, sources_prioritization: List[str] = None
):
    """
    Since there is ambiguity in the information, the following filtering scheme was decided upon:
    •	For each InChI, check how many of the sources agree on it.
    •	If there is one that has the most votes, trust it as the correct answer.
    •	If there are multiple results with the same number of votes, weight the votes
        according to the sources prioritization
    •	If there is a result with the highest score, trust it as the correct answer.
    •	If there is still ambiguity, reject the drug name and do not trust any result.
    Args:
        inchi_dict: dictionary of inchi lists from different sources
        sources_prioritization: list of sources prioritization
    Returns:
        - The best choice of inchi
        - The source of the best choice
        If there is no best choice, returns None, []
    """
    if sources_prioritization is None:
        sources_prioritization = PRIORITIZED_SOURCES
    sources_prioritization = [source + " inchi" for source in sources_prioritization]

    all_inchis = set()
    inchi_dict = inchi_dict.copy()
    for source in sources_prioritization:
        if inchi_dict[source] is None:
            inchi_dict[source] = set()
        else:
            inchi_dict[source] = set(inchi_dict[source])
        all_inchis = all_inchis | inchi_dict[source]
    inchis_and_votes = [
        (
            inchi,
            [
                source
                for source in sources_prioritization
                if inchi in inchi_dict[source]
            ],
        )
        for inchi in all_inchis
    ]
    if not inchis_and_votes:
        return None, []
    inchis_and_votes = sorted(
        inchis_and_votes, key=lambda x: -len(x[1])
    )  # most votes first
    most_votes = len(inchis_and_votes[0][1])
    inchis_with_most_votes = [
        couple for couple in inchis_and_votes if len(couple[1]) == most_votes
    ]
    if len(inchis_with_most_votes) == 1:
        best_inchi, sources_that_agree_on_best_inchi = inchis_with_most_votes[0]
    else:
        # We need to prioritise using the sources prioritization
        reverse_prioritization = list(reversed(sources_prioritization))
        inchis_with_scores = [
            (
                inchi,
                sources_of_inchi,
                sum(
                    [
                        reverse_prioritization.index(source) + 1
                        for source in sources_of_inchi
                    ]
                ),
            )
            for inchi, sources_of_inchi in inchis_with_most_votes
        ]
        inchis_with_scores = sorted(inchis_with_scores, key=lambda x: -x[2])
        most_votes = inchis_with_scores[0][2]
        inchis_with_highest_score = [
            triplet for triplet in inchis_with_scores if triplet[2] == most_votes
        ]
        if len(inchis_with_highest_score) == 1:
            best_inchi, sources_that_agree_on_best_inchi = (
                inchis_with_highest_score[0][0],
                inchis_with_highest_score[0][1],
            )
        else:  # Then we cannot make a decision
            return None, []
    return best_inchi, sources_that_agree_on_best_inchi


def get_inchis_and_smiles_from_all_sources(
    names: List[str], sources_list: List[str] = None
) -> (pd.DataFrame, dict):
    """
    Creates a dataframe that includes all InChI strings found from the given sources. A voting scheme according
    to the prioritization is used to chose then "correct" InChI where it is ambiguous.
    Args:
        names: A list of drug names
        sources_list: A list of sources to use. If None, all sources will be used.
    Returns:
        A dataframe which, for every source, lists the InChI strings found for each drug name.
        A dict of sources paired with a list of tuples of (drug name, exception) for the drugs for which no
        InChI was found using that source.
    """
    if sources_list is None:
        sources_list = PRIORITIZED_SOURCES

    assert all(
        [s in PRIORITIZED_SOURCES for s in sources_list]
    ), f"All sources must be in {PRIORITIZED_SOURCES}!"

    rows = []
    failed = {source_name: [] for source_name in PRIORITIZED_SOURCES}
    for name in tqdm(names, "Gathering inchi from sources"):
        row = dict()
        row["name"] = name

        # Gather inchis from sources
        for source in sources_list:
            try:
                inchi_list = list(set(inchi_functions_dict[source](name)))
                smiles_list = [inchi_to_canonical_smiles(inchi) for inchi in inchi_list]
            except InChINotFoundException as e:
                inchi_list = None
                smiles_list = None
                failed[source].append((name, e))
            row[f"{source} inchi"] = inchi_list
            row[f"{source} smiles"] = smiles_list
        rows.append(row)

    df = pd.DataFrame(rows)
    return df, failed


def add_inchi_and_smiles_to_df(
    df: pd.DataFrame, name_col: str = "name",
) -> (pd.DataFrame, dict):
    """
    Adds the InChI and SMILES columns to the given dataframe, using the prioritized sources and the voting scheme
    (see helper functions for more details).
    Args:
        df: A dataframe with a "name" column, which is a column on strings.
        name_col: The name of the column in the dataframe that contains the drug names.
    Returns:
        - The same dataframe, with the "inchi" and "smiles" columns added. If they are None, it means either that
          no inchis could be found or that even though inchis were found, no one inchi could be chosen as the best one.
        - A dict of sources paired with a list of tuples of (drug name, exception) for the drugs for which no
        InChI was found using that source.
    """
    df = df.copy()
    for c in ["inchi", "smiles"]:
        if c in df.columns:
            print(
                f"Column {c} already exists in the dataframe! It will be overwritten."
            )
            df = df.drop(columns=c)
    df["inchi"] = None
    df["smiles"] = None
    df_for_sources = df[df["inchi"].isnull()]
    if len(df_for_sources) == 0:
        return df, {}
    inchis_and_smiles_from_all_df, failed = get_inchis_and_smiles_from_all_sources(
        df_for_sources[name_col]
    )
    dict_for_new_df = {name_col: inchis_and_smiles_from_all_df["name"].values}
    inchis = []
    smiles = []
    for _, row in inchis_and_smiles_from_all_df.iterrows():
        row_without_name = row.drop("name")
        inchi_dict = row_without_name.to_dict()
        best_inchi, _ = get_best_choice_of_inchi(inchi_dict)
        best_smiles = (
            None if best_inchi is None else inchi_to_canonical_smiles(best_inchi)
        )
        inchis.append(best_inchi)
        smiles.append(best_smiles)
    dict_for_new_df["inchi"] = inchis
    dict_for_new_df["smiles"] = smiles
    new_df = pd.DataFrame(dict_for_new_df)

    df_for_sources = df_for_sources.drop(columns=["inchi", "smiles"])
    df_for_sources = df_for_sources.merge(new_df, on=name_col, how="left")
    df = df[~df["inchi"].isnull()]
    df = pd.concat([df, df_for_sources])
    return df, failed


def get_inchi_and_smiles_from_list_of_names(
    names: List[str]
) -> (List[str], List[str], dict):
    """
    Gets the InChI and SMILES strings for the given list of drug names.
    Args:
        names: A list of drug names
    Returns:
        - A list of the corresponding InChI strings
        - A list of the corresponding SMILES strings
        - A dict of sources paired with a list of tuples of (drug name, exception) for the drugs for which no
        InChI was found using that source.
    """
    names_df = pd.DataFrame({"name": names})
    names_df, failed = add_inchi_and_smiles_to_df(
        names_df, "name",
    )

    names_df = names_df.reset_index()
    names_df = (
        names_df.set_index("name")
        .reindex(index=names)
        .reset_index()
        .drop(columns=["index"])
    )

    inchis = names_df["inchi"].values
    smiles = names_df["smiles"].values
    return inchis, smiles, failed
