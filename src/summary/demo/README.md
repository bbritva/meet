# Démo — qualité des transcriptions : avant / après

Deux fonctionnalités, un seul transcript, deux documents La Suite Docs.

| | AVANT | APRÈS |
|---|---|---|
| `is_resolve_speaker_cues_enabled` | `False` | `True` |
| `is_acronym_correction_enabled` | `False` | `True` |
| `resolve_speaker_cues_detector` | — | `llm` |

L'« avant » n'est pas écrit à la main. C'est ce que fait `celery_worker.py`
quand les deux drapeaux sont fermés : aucune des deux étapes n'est appelée et
`format_transcript` tourne sur le JSON WhisperX brut. C'est le comportement
d'aujourd'hui pour Dictaphone, qui n'a jamais de métadonnées VAD — d'où les
`SPEAKER_00` du document brut.

Le script appelle les fonctions de `celery_worker.py`, dans son ordre :
`resolve_speaker_identities_and_apply_to` → `_correct_acronyms_in` →
`format_transcript`, puis `create_document_in_lasuite_docs`. Il ne réimplémente
rien et ne modifie ni les deux fonctionnalités, ni `docs_service.py`.

La démo vit dans `src/summary/demo/` et non à la racine du dépôt, parce que
seul `./src/summary` est monté dans les conteneurs (`/app`) : c'est la seule
place d'où elle peut tourner dans le vrai environnement du service.

| | |
|---|---|
| `run_demo.py` | le script : deux passages, les preuves, le score, la publication |
| `llm_cache.py` | cache disque devant `LLMService`, pour rejouer hors ligne |
| `input/` | copies en lecture seule des mocks (`mocks/flow-1-technique`, `speaker-attribution/cases/06-no-cues`) |
| `cache/` | les réponses d'Albert, **versionnées** : sans elles la démo dépend du wifi de la salle |
| `out/` | les markdown poussés dans Docs et les rapports |

## Comment le rejouer

```sh
docker exec -e IS_ACRONYM_CORRECTION_ENABLED=true \
            -e IS_RESOLVE_SPEAKER_CUES_ENABLED=true \
            -e RESOLVE_SPEAKER_CUES_DETECTOR=llm \
            -e PYTHONHASHSEED=0 \
            celery-summary-transcribe python /app/demo/run_demo.py
```

`--offline` interdit tout appel réseau (le cache doit suffire), `--no-docs`
saute la publication.

`PYTHONHASHSEED=0` n'est pas décoratif : `phonetic.PhoneticIndex` range les
acronymes dans un `set`, et `candidates()` départage les ex æquo par ordre
d'itération. Sans graine fixée, la liste de candidats — donc le prompt
`acronym-decide` — change d'un processus à l'autre et le cache disque rate.
La graine ne change aucune décision, elle enlève seulement l'aléa qui
empêcherait de rejouer la démo hors ligne.


## Honnêteté — à lire avant tout le reste

- **La transcription et la liste des participants sont fabriquées.** Le fichier
  WhisperX, l'invitation `.ics` et les quatre noms sont des mocks écrits pour
  cette démonstration. Aucune vraie réunion n'est passée dans ce pipeline.
- Les erreurs mesurées sont **nos propres erreurs**, écrites dans `errors.json`
  avant la mesure. Ce sont des chiffres synthétiques, pas des résultats de
  terrain.
- Le rappel de 0,74 du résolveur de locuteurs est une **estimation ponctuelle** :
  `temperature=0` n'est pas reproductible sur Albert, les répétitions donnent le
  même score mais sur des items différents.
- Un nombre de tests unitaires n'est pas une mesure de justesse, et n'est pas
  présenté comme telle ici.
- Le modèle est celui que la pile a déjà configuré : `LLM_MODEL=openweight-large`
  chez Albert (`openai/gpt-oss-120b`). La démo n'en choisit pas un autre pour
  l'occasion.
- Les deux documents publiés viennent d'un passage **entièrement en cache**,
  sans un seul appel réseau. Les réponses en cache, elles, ont bien été
  produites par Albert lors d'un passage précédent sur cette même entrée.

## 1. Les deux documents

- **AVANT** — Réunion architecture — transcript brut
  http://localhost:8700/docs/9142519d-5ded-44de-a085-d411de066429/
- **APRÈS** — Réunion architecture — transcript corrigé
  http://localhost:8700/docs/ffd96509-0ec9-4a72-abff-988987de653b/

Captures d'écran des deux documents ouverts dans Docs : `/Users/macair/dinum/screenshots/demo-docs-transcript-brut-v2.png` et `…-corrige-v2.png`.

Markdown correspondant, tel qu'il a été poussé : `demo/out/flow-1-before.md` et `demo/out/flow-1-after.md`.

## 2. Preuve par nom (résolveur d'indices, détecteur `llm`)

Format : `Nom ← type d'indice, « citation », segment`

```
Julien Perrot ← self_id, « Bonjour, Julien à l'appareil. », segment 5 (confiance 0.95, label SPEAKER_01)
Karim Sahli ← self_id, « Karim, bonjour à tous. », segment 17 (confiance 0.95, label SPEAKER_03)
Mathilde Roux ← self_id, « Mathilde, bonjour. », segment 12 (confiance 0.95, label SPEAKER_02)
Nadia Berger ← self_id, « Ici Nadia. », segment 1 (confiance 0.95, label SPEAKER_00)
```

Labels laissés en `SPEAKER_XX` dans le document corrigé : **aucun**

Contrôle contre la vérité terrain — le champ `speaker` de `errors.json` dit qui a réellement parlé sur chaque segment fautif. **3 / 3** labels retrouvés correctement. Le contrôle ne couvre que 3 des 4 labels : `errors.json` est une vérité terrain d'acronymes, pas de locuteurs, et un participant qui n'a commis aucune erreur d'acronyme n'y figure pas. Les 4 noms du document corrigé sont néanmoins les 4 bons.

| label WhisperX | attendu | obtenu | |
|---|---|---|---|
| SPEAKER_01 | Julien Perrot | Julien Perrot | ✅ |
| SPEAKER_02 | Mathilde Roux | Mathilde Roux | ✅ |
| SPEAKER_03 | Karim Sahli | Karim Sahli | ✅ |

## 3. Preuve par acronyme

Format : `ACRONYME ← "ce que Whisper a écrit", confiance, appliqué ou non`

### Appliqués (confiance ≥ 0.80)

```
Yjs ← "y grec js", confidence 0.95, applied [segment 6, word 8]
CRDT ← "serre des thés", confidence 0.95, applied [segment 7, word 7]
MinIO ← "mini eau", confidence 0.85, applied [segment 17, word 9]
WOPI ← "whoopee", confidence 0.85, applied [segment 21, word 7]
Keycloak ← "qui cloaque", confidence 0.95, applied [segment 26, word 4]
OIDC ← "oh idée sait", confidence 0.95, applied [segment 27, word 6]
Menshen ← "main chêne", confidence 0.85, applied [segment 30, word 3]
```

### Sous le seuil — signalés mais **NON appliqués**

C'est l'histoire de la précision : la correction est trouvée, elle est remontée dans la liste d'audit, et le document n'est pas modifié.

```
DINUM ← "dix nomme", confidence 0.75, NOT applied (below floor) [segment 5, word 5]
```

## 4. Score contre `errors.json`

Vérité terrain : **15 erreurs** au total, dont **12 erreurs d'acronyme** (le périmètre de la fonctionnalité) et **3 hors périmètre**.

| Résultat | Nombre |
|---|---|
| corrigées (appliquées, bonne réponse) | **7 / 12** |
| bonne réponse mais sous le seuil (non appliquée) | 1 |
| manquées | 4 |
| faux positifs | 0 |

### Détail, erreur par erreur

| id | Whisper a écrit | attendu | résultat |
|---|---|---|---|
| e1 | dix nomme | DINUM | trouvée mais sous le seuil (0.75) — non appliquée |
| e2 | y grec js | Yjs | corrigée (0.95) |
| e3 | serre des thés | CRDT | corrigée (0.95) |
| e5 | mini eau | MinIO | corrigée (0.85) |
| e6 | whoopee | WOPI | corrigée (0.85) |
| e7 | qui cloaque | Keycloak | corrigée (0.95) |
| e8 | oh idée sait | OIDC | corrigée (0.95) |
| e9 | main chêne | Menshen | corrigée (0.85) |
| e11 | type est | Typst | manquée |
| e13 | dos pecs | Docspec | manquée |
| e14 | bloc note | BlockNote | manquée |
| e15 | Christ | Grist | manquée |

### Ce qui a été manqué, et pourquoi

- **`dix nomme` → DINUM (e1)** est **trouvé** à 0.75 et remonté dans la liste d'audit, mais sous le seuil de 0.80 : le document garde le mot de Whisper. C'est un changement de comportement : au passage précédent il était **manqué**, parce que la fenêtre élargie « Côté dix nomme » matchait COTRIM (0.70), réclamait les mots et bloquait « dix nomme » → DINUM. `WINDOW_MARGIN = 0` (commit 28729b37) supprime cette fenêtre que personne n'avait signalée, et le bon candidat atteint l'arbitrage — sans convaincre le modèle pour autant. Corrigé, il ne l'est toujours pas.
- **`type est` → Typst (e11)**, **`dos pecs` → Docspec (e13)**, **`bloc note` → BlockNote (e14)**, **`Christ` → Grist (e15)** : ces entrées **sont** pourtant dans le glossaire de 7769 entrées. Le blocage ne vient donc pas d'un candidat manquant : soit l'étape 1 n'a pas signalé le passage, soit le modèle a refusé à l'étape 2. La démo n'instrumente pas l'étape 1 et ne tranche pas entre les deux.
- **Zéro faux positif** sur ce transcript. C'est le sens du compromis : le seuil de 0.80 est réglé pour la précision, pas pour le rappel.

### Hors périmètre — la correction d'acronymes ne les vise pas

| id | Whisper a écrit | attendu | type |
|---|---|---|---|
| e4 | treize | trente | number |
| e10 | dix-huit | dix-sept | number |
| e12 | Marie-Anne | Marianne | proper_noun |

Ce sont deux nombres (`treize`→`trente`, `dix-huit`→`dix-sept`) et un nom propre (`Marie-Anne`→`Marianne`). Le glossaire ne contient que des acronymes administratifs : rien dans cette fonctionnalité ne peut les rattraper, et il ne faut pas le lui reprocher.

## 5. Le refus — cas `06-no-cues`

La question qui vient toujours : « comment vous évitez de mettre des mots dans la bouche de quelqu'un ? » Réponse : sur une réunion sans aucun indice de nom, le résolveur n'attribue rien.

- participants invités : Damien Laroche, Fabienne Roussel, Ingrid Hoffmann, Pascal Nourry
- labels dans la transcription : SPEAKER_00, SPEAKER_01, SPEAKER_02
- noms attribués : **aucun**
- labels laissés en `SPEAKER_XX` : **SPEAKER_00, SPEAKER_01, SPEAKER_02**

Ce que le modèle a vu, et ce qui a été refusé :

```
cue mention  segment 2   speaker SPEAKER_00  « Damien Laroche » -> Damien Laroche
```

Le piège est le segment 2, « comme Damien l'avait signalé » : une mention à la troisième personne. `resolve.py` ne transforme **jamais** une mention en attribution. La vérité terrain dit que SPEAKER_00 est réellement Fabienne Roussel — et que ne pas l'attribuer est la bonne réponse.

## 6. Reproductibilité hors ligne

Toutes les réponses d'Albert sont en cache sur disque (`demo/cache/llm-cache.json`, clé = hash du modèle et des deux prompts). Le second passage est rejoué avec `--offline`, qui transforme tout défaut de cache en erreur :

```
passage 1 : 31 réponses lues en cache, 0 appels à Albert
passage 2 (--offline, tout défaut de cache = erreur) : 31 en cache, 0 appels
markdown identique entre les deux passages : oui
```

## 7. Les lignes de journal qui prouvent que les étapes ont tourné

```
Speaker resolution for task demo-after: source=cues (no usable metadata), 4 assigned, 0 unassigned
Acronym correction: 8 correction(s) found, 7 applied
Acronym correction for task demo-after: 8 correction(s), 7 applied
Speaker resolution for task demo-after: source=cues (no usable metadata), 4 assigned, 0 unassigned
Acronym correction: 8 correction(s) found, 7 applied
Acronym correction for task demo-after: 8 correction(s), 7 applied
Speaker resolution for task demo-refusal: source=cues (no usable metadata), 0 assigned, 3 unassigned
```

## 8. Variance entre deux passages identiques

Même entrée, même code, même seuil — seule la génération d'Albert change. `temperature=0` n'est pas reproductible chez eux : ce tableau est la preuve, pas une excuse écrite après coup.

| passage | corrigées | sous le seuil | manquées | faux positifs |
|---|---|---|---|---|
| passage de référence (celui publié) | 7 | 1 | 4 | 0 |
| report-run2.md | 6 | 2 | 4 | 0 |

Corrections sous le seuil observées, passage par passage :

```
passage de référence (celui publié):
  DINUM ← "dix nomme", confidence 0.75, NOT applied (below floor) [segment 5, word 5]
report-run2.md:
  DINUM ← "dix nomme", confidence 0.75, NOT applied (below floor) [segment 5, word 5]
  Grist ← "Christ", confidence 0.75, NOT applied (below floor) [segment 46, word 7]
```

C'est exactement ce que le seuil est censé faire : quand le modèle n'est pas sûr, la correction est **remontée dans la liste d'audit** et le document reste tel quel. Une erreur laissée en place se corrige à la relecture ; un mot inventé, non.
