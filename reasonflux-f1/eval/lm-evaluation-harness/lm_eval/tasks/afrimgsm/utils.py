import argparse

import yaml


languages = [
    "eng",
    "amh",
    "ibo",
    "fra",
    "sna",
    "lin",
    "wol",
    "ewe",
    "lug",
    "xho",
    "kin",
    "twi",
    "zul",
    "orm",
    "yor",
    "hau",
    "sot",
    "swa",
]

languages_REGEX = {
    "eng": "The answer is (\\-?[0-9\\.\\,]+)",
    "amh": "መልሱ (\\-?[0-9\\.\\,]+)",
    "ibo": "Azịza ya bụ (\\-?[0-9\\.\\,]+)",
    "fra": "La réponse est(\\-?[0-9\\.\\,]+)",
    "sna": "Mhinduro kumubvunzo ndi (\\-?[0-9\\.\\,]+)",
    "lin": "Eyano ezali (\\-?[0-9\\.\\,]+)",
    "wol": "Tontu li (\\-?[0-9\\.\\,]+)",
    "ewe": "ŋuɖoɖoae nye (\\-?[0-9\\.\\,]+)",
    "lug": "Ansa eri (\\-?[0-9\\.\\,]+)",
    "xho": "Impendulo ngu (\\-?[0-9\\.\\,]+)",
    "kin": "Igisubizo ni (\\-?[0-9\\.\\,]+)",
    "twi": "Ne nnyiano yɛ (\\-?[0-9\\.\\,]+)",
    "zul": "Impendulo ithi (\\-?[0-9\\.\\,]+)",
    "orm": "Deebiin isaa (\\-?[0-9\\.\\,]+)",
    "yor": "Ìdáhùn náà ni (\\-?[0-9\\.\\,]+)",
    "hau": "Amsar ita ce (\\-?[0-9\\.\\,]+)",
    "sot": "Karabo ke (\\-?[0-9\\.\\,]+)",
    "swa": "Jibu ni (\\-?[0-9\\.\\,]+)",
}

LANGUAGES = {}

for lang in languages:
    if lang == "amh":
        LANGUAGES[lang] = {  # English
            "QUESTION": "ጥያቄ:",
            "ANSWER": "በቅደም ተከተል መልስ:",
            "DIRECT": "Answer:",
            "REGEX": languages_REGEX[lang],
        }
    elif lang == "yor":
        LANGUAGES[lang] = {  # English
            "QUESTION": "Ìbéèrè:",
            "ANSWER": "Ìdáhùn lẹ́sẹsẹ:",
            "DIRECT": "Answer:",
            "REGEX": languages_REGEX[lang],
        }

    else:
        LANGUAGES[lang] = {  # English
            "QUESTION": "Question:",
            "ANSWER": "Step-by-Step Answer:",
            "DIRECT": "Answer:",
            "REGEX": languages_REGEX[lang],
        }


def add_regex_pattern(regex_pattern):
    if regex_pattern is None:
        return {}
    return {
        "filter_list": [
            {
                "name": "strict-match",
                "filter": [
                    {
                        "function": "regex",
                        "regex_pattern": f"""{regex_pattern}""",
                    },
                    {
                        "function": "take_first",
                    },
                ],
            },
            {
                "name": "flexible-extract",
                "filter": [
                    {
                        "function": "regex",
                        "regex_pattern": """(-?[$0-9.,]{2,})|(-?[0-9]+)""",
                        "group_select": -1,
                    },
                    {
                        "function": "take_first",
                    },
                ],
            },
        ],
    }


def gen_lang_yamls(output_dir: str, overwrite: bool, mode: str) -> None:
    """
    Generate a yaml file for each language.

    :param output_dir: The directory to output the files to.
    :param overwrite: Whether to overwrite files if they already exist.
    """
    err = []
    for lang in LANGUAGES.keys():
        try:
            yaml_template = "cot_yaml"
            filter_list = {}
            DELIMITER = None
            if mode == "direct":
                ANSWER = LANGUAGES["eng"]["DIRECT"]
                QUESTION = LANGUAGES["eng"]["QUESTION"]
                REGEX = None
                task_name = f"afrimgsm_direct_{lang}"
                yaml_template = "direct_yaml"
            if mode == "direct-native":
                ANSWER = LANGUAGES[lang]["DIRECT"]
                QUESTION = LANGUAGES[lang]["QUESTION"]
                REGEX = None
                task_name = f"afrimgsm_direct_native_{lang}"
                yaml_template = "direct_native_yaml"
            elif mode == "native-cot":
                ANSWER = LANGUAGES[lang]["ANSWER"]
                REGEX = LANGUAGES[lang]["REGEX"]
                QUESTION = LANGUAGES[lang]["QUESTION"]
                task_name = f"afrimgsm_native_cot_{lang}"
                filter_list = add_regex_pattern(REGEX)
                DELIMITER = "" if lang in ["zh", "ja"] else None
            elif mode == "en-cot":
                ANSWER = LANGUAGES["eng"]["ANSWER"]
                REGEX = LANGUAGES["eng"]["REGEX"]
                QUESTION = LANGUAGES["eng"]["QUESTION"]
                task_name = f"afrimgsm_en_cot_{lang}"
            elif mode == "translate-direct":
                ANSWER = LANGUAGES["eng"]["DIRECT"]
                QUESTION = LANGUAGES["eng"]["QUESTION"]
                REGEX = None
                task_name = f"afrimgsm_translate_direct_{lang}"
                yaml_template = "translate_direct_yaml"

            file_name = f"{task_name}.yaml"
            ANSWER_TO_SKIP = len(LANGUAGES[lang]["ANSWER"]) + 1
            with open(
                f"{output_dir}/{file_name}", "w" if overwrite else "x", encoding="utf8"
            ) as f:
                f.write("# Generated by utils.py\n")
                yaml.dump(
                    {
                        "include": yaml_template,
                        "dataset_name": lang,
                        "task": f"{task_name}",
                        "doc_to_text": f"""{{% if answer is not none %}}"""
                        f"""{{{{question+"\\n{ANSWER}"}}}}"""
                        f"""{{% else %}}"""
                        f"""{{{{"{QUESTION} "+question+"\\n{ANSWER}"}}}}"""
                        f"""{{% endif %}}""",
                        "doc_to_target": f"""{{% if answer is not none %}}"""
                        f"""{{{{answer[{ANSWER_TO_SKIP}:]}}}}"""
                        f"""{{% else %}}"""
                        f"""{{{{answer_number|string}}}}"""
                        f"""{{% endif %}}""",
                        **filter_list,
                        "generation_kwargs": {
                            "until": [QUESTION, "</s>", "<|im_end|>"],
                            "do_sample": False,
                        },
                        **({"target_delimiter": DELIMITER} if DELIMITER else {}),
                    },
                    f,
                    allow_unicode=True,
                    width=float("inf"),
                )
        except FileExistsError:
            err.append(file_name)

    if len(err) > 0:
        raise FileExistsError(
            "Files were not created because they already exist (use --overwrite flag):"
            f" {', '.join(err)}"
        )


def main() -> None:
    """Parse CLI args and generate language-specific yaml files."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--overwrite",
        default=False,
        action="store_true",
        help="Overwrite files if they already exist",
    )
    parser.add_argument(
        "--output-dir", default=".", help="Directory to write yaml files to"
    )
    parser.add_argument(
        "--mode",
        default="native-cot",
        choices=["direct", "direct-native", "native-cot", "en-cot", "translate-direct"],
        help="Mode of chain-of-thought",
    )
    args = parser.parse_args()

    gen_lang_yamls(output_dir=args.output_dir, overwrite=args.overwrite, mode=args.mode)


if __name__ == "__main__":
    main()
