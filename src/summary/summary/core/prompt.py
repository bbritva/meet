# ruff: noqa

PROMPT_SYSTEM_TLDR = """Tu es un agent dont le rôle est de créer un TL;DR (résumé très concis) d'un compte rendu de réunion. Tu utiliseras un style synthétique, administratif, à la troisième personne, sans affect. Tu recevras en entrée le transcript. Ta tâche est de rédiger un résumé concis et structuré, en te concentrant uniquement sur les informations essentielles et pertinentes. Tu répondras en un paragraphe structuré (3 à 6 phrases), sans rien ajouter d'autre. Tu répondras dans le format suivant sans rien ajouter d'autre:
### Résumé TL;DR
[Résumé concis et structuré]"""

PROMPT_SYSTEM_PLAN = """Ta tâche est de diviser le contenu du transcript en sujets concrets correspondant aux grands axes discutés durant la réunion. Ne crée pas de catégories génériques. Les titres doivent être courts, précis et représentatifs des échanges. Veille à ce que chaque sujet soit distinct et qu’aucun thème ne soit répété. Tu te limiteras à 5 ou 6 sujets maximum. 
L'introduction, ordre du jour, conclusion, etc. seront rajoutés a posteriori. Si il n'y a pas de sujets clairs, réponds "Général".
"""

PROMPT_SYSTEM_PART = """Tu es un agent dont le rôle est de créer une partie du résumé d'un compte rendu de réunion. Tu utiliseras un style synthétique, administratif, à la troisième personne, sans affect. Tu recevras en entrée le transcript, et le titre du sujet correspondant. Ta tâche est de rédiger un résumé concis de cette partie et uniquement cette partie, en te concentrant uniquement sur les informations essentielles et pertinentes. Le résumé de chaque partie doit tenir en 4 à 6 phrases maximum, sans entrer dans les détails mineurs. Tu répondras dans le format suivant :
    ### Titre du sujet [Traduire ce titre selon la langue du transcript]
    [Résumé concis et structuré de la partie du transcript]
    """

PROMPT_USER_PART = """Titre de la partie à résumer : {part}
Transcript complet :
{transcript}"""

PROMPT_SYSTEM_CLEANING = """Tu es un agent dont le rôle est de nettoyer un résumé de compte rendu de réunion. Tu recevras en entrée le résumé brut, potentiellement avec des erreurs de formatage, des incohérences ou des redondances. Ta tâche est de corriger les erreurs de formatage, d'améliorer la clarté et la cohérence du texte, et de t'assurer que le résumé est bien structuré et facile à lire. Ton but principal est de retirer les redondances et les répétitions. Assure la cohérence entre les titres et homogénéise le style d’écriture entre les parties. Supprime les doublons d’informations entre les parties si présents. Si certaines parties sont plus secondaires, tu peux les fusionner ou les réduire en 1 à 2 phrases. Mets en avant les points centraux qui ont fait l’objet de décisions ou d’actions. Tu répondras uniquement avec le résumé sans rien ajouter d'autre"""

PROMPT_SYSTEM_NEXT_STEP = """Tu es un agent dont le rôle est d'extraire les prochaines étapes d'un transcript de réunion. Tu utiliseras un style synthétique, administratif, à la troisième personne, sans affect. Tu recevras en entrée le transcript. Ta tâche est d'identifier et de lister toutes les actions à entreprendre, en indiquant la ou les personnes assignées et en précisant les échéances si elles sont mentionnées. Ne retiens que les actions concrètes et à venir. Ignore les remarques générales ou les constats sans suite."""

FORMAT_NEXT_STEPS = {
    "type": "json_schema",
    "json_schema": {
        "name": "actions",
        "schema": {
            "type": "object",
            "properties": {
                "actions": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "title": {"type": "string"},
                            "assignees": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Noms des personnes assignées",
                            },
                            "due_date": {
                                "type": "string",
                                "description": "Date d'échéance si mentionnée (si l'année nest pas précisée, ne pas l'ajouter)",
                            },
                        },
                        "required": ["title", "assignees"],
                        "additionalProperties": False,
                    },
                }
            },
            "required": ["actions"],
            "additionalProperties": False,
        },
        "strict": True,
    },
}

FORMAT_PLAN = {
    "type": "json_schema",
    "json_schema": {
        "name": "Titles",
        "schema": {
            "type": "object",
            "properties": {"titles": {"type": "array", "items": {"type": "string"}}},
            "required": ["titles"],
            "additionalProperties": False,
        },
        "strict": True,
    },
}

PROMPT_SYSTEM_ACRONYM_DETECT = """Tu analyses la transcription automatique d'une réunion administrative française. Le modèle de reconnaissance vocale a remplacé des sigles qu'il ne connaissait pas par des mots français courants qui SONNENT pareil mais n'ont aucun sens ici (ex : "dix nomme" au lieu de "DINUM", "quenil" au lieu de "CNIL"). Ta tâche est de repérer les passages qui NE FONT PAS SENS. Tu ne corriges rien.
RÈGLE ESSENTIELLE : "span" doit contenir UNIQUEMENT les mots suspects, de 1 à 4 mots maximum, copiés exactement. Jamais la phrase entière, jamais de ponctuation autour.
Exemple correct : {"suspects":[{"span":"dix nomme"}]}
Exemple interdit : {"suspects":[{"span":"Côté dix nomme, la brique a avancé."}]}
Tu répondras uniquement en JSON, sans rien ajouter d'autre : {"suspects":[{"span":"<1 à 4 mots>"}]}"""

PROMPT_SYSTEM_ACRONYM_DECIDE = """Tu corriges la transcription automatique d'une réunion administrative française. Un passage est suspect : la reconnaissance vocale a peut-être mal transcrit un sigle. Tu recevras la phrase, le passage suspect, et une liste de sigles qui SONNENT pareil. Ta tâche est de choisir le sigle qui correspond vraiment, en t'appuyant sur le SENS de la phrase. Si aucun ne convient, ou si le passage est correct tel quel, tu répondras null. Dans le doute, tu répondras null : une correction fausse est pire qu'une correction manquée. Tu indiqueras ta confiance entre 0.0 et 1.0.
Tu répondras uniquement en JSON, sans rien ajouter d'autre : {"choix": "<SIGLE>" ou null, "confiance": 0.0}"""

PROMPT_USER_ACRONYM_DECIDE = """Phrase : « {sentence} »
Passage suspect : « {span} »

Sigles candidats :
{candidates}"""

FORMAT_ACRONYM_DETECT = {
    "type": "json_schema",
    "json_schema": {
        "name": "suspects",
        "schema": {
            "type": "object",
            "properties": {
                "suspects": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "span": {
                                "type": "string",
                                "description": "Passage suspect, 1 à 4 mots",
                            }
                        },
                        "required": ["span"],
                        "additionalProperties": False,
                    },
                }
            },
            "required": ["suspects"],
            "additionalProperties": False,
        },
        "strict": True,
    },
}

FORMAT_ACRONYM_DECIDE = {
    "type": "json_schema",
    "json_schema": {
        "name": "acronym_choice",
        "schema": {
            "type": "object",
            "properties": {
                "choix": {
                    "type": ["string", "null"],
                    "description": "Sigle retenu, ou null si aucun ne convient",
                },
                "confiance": {"type": "number"},
            },
            "required": ["choix", "confiance"],
            "additionalProperties": False,
        },
        "strict": True,
    },
}
