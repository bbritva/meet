## 1. Les deux documents

- **AVANT** — Réunion architecture — transcript brut
  http://localhost:8700/docs/1ed36d67-c730-4d6a-b26e-b58556966382/
- **APRÈS** — Réunion architecture — transcript corrigé
  http://localhost:8700/docs/787d416f-38fc-4d06-a4b2-a92cd7899d69/

Captures d'écran des deux documents ouverts dans Docs : `/Users/macair/dinum/screenshots/demo-docs-transcript-brut.png` et `…-corrige.png`.

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
CRDT ← "des serre des thés", confidence 0.90, applied [segment 7, word 6]
MinIO ← "dans mini eau mais", confidence 0.86, applied [segment 17, word 8]
WOPI ← "whoopee", confidence 0.85, applied [segment 21, word 7]
Keycloak ← "qui cloaque", confidence 0.95, applied [segment 26, word 4]
OIDC ← "oh idée sait", confidence 0.93, applied [segment 27, word 6]
Menshen ← "main chêne qui", confidence 0.85, applied [segment 30, word 3]
```

### Sous le seuil — signalés mais **NON appliqués**

C'est l'histoire de la précision : la correction est trouvée, elle est remontée dans la liste d'audit, et le document n'est pas modifié.

```
(aucun)
```

## 4. Score contre `errors.json`

Vérité terrain : **15 erreurs** au total, dont **12 erreurs d'acronyme** (le périmètre de la fonctionnalité) et **3 hors périmètre**.

| Résultat | Nombre |
|---|---|
| corrigées (appliquées, bonne réponse) | **6 / 12** |
| bonne réponse mais sous le seuil (non appliquée) | 0 |
| manquées | 6 |
| faux positifs | 0 |

### Détail, erreur par erreur

| id | Whisper a écrit | attendu | résultat |
|---|---|---|---|
| e1 | dix nomme | DINUM | manquée |
| e2 | y grec js | Yjs | manquée |
| e3 | serre des thés | CRDT | corrigée (0.90) |
| e5 | mini eau | MinIO | corrigée (0.86) |
| e6 | whoopee | WOPI | corrigée (0.85) |
| e7 | qui cloaque | Keycloak | corrigée (0.95) |
| e8 | oh idée sait | OIDC | corrigée (0.93) |
| e9 | main chêne | Menshen | corrigée (0.85) |
| e11 | type est | Typst | manquée |
| e13 | dos pecs | Docspec | manquée |
| e14 | bloc note | BlockNote | manquée |
| e15 | Christ | Grist | manquée |

### Effet de bord observé : la fenêtre mange des mots voisins

La même règle de « plus longue fenêtre d'abord » qui fait manquer DINUM fait aussi remplacer plus de mots que nécessaire. Le mot juste arrive, mais la phrase perd un mot autour. À regarder avant toute mise en production — ce n'est pas une invention de contenu, c'est une phrase abîmée.

| attendu | fenêtre réellement remplacée | phrase obtenue |
|---|---|---|
| `serre des thés` → CRDT | `des serre des thés` → CRDT | Le modèle de données repose sur CRDT, donc la fusion des modifications se fait sans verrou côté serveur. |
| `mini eau` → MinIO | `dans mini eau mais` → MinIO | Karim, bonjour à tous. Les deux applications écrivent MinIO avec deux conventions de nommage différentes. |
| `main chêne` → Menshen | `main chêne qui` → Menshen | On passe par Menshen implémente l'échange de jetons décrit dans la RFC 8693. |

### Ce qui a été manqué, et pourquoi

- **`dix nomme` → DINUM (e1)** est manqué, et c'est un bug connu, documenté avant cette démo (`BRIEF-demo-agent-acronyms.md` §10) : la fenêtre de 3 mots « Côté dix nomme » ressemble à COTRIM (0.70) et, parce que les fenêtres sont réclamées de la plus longue à la plus courte, elle bloque la fenêtre de 2 mots qui aurait donné DINUM. Le seuil de confiance empêche COTRIM de passer, donc le document reste juste — mais l'exemple phare ne se corrige pas.
- **`y grec js` → Yjs (e2)**, **`type est` → Typst (e11)**, **`dos pecs` → Docspec (e13)**, **`bloc note` → BlockNote (e14)** : ces noms de logiciels libres ne sont pas dans le glossaire administratif de 7769 entrées. Rien à décider si le candidat n'existe pas.
- **`Christ` → Grist (e15)** dépend du passage : voir la section variance plus bas.
- **Zéro faux positif** sur ce transcript. C'est le sens du compromis : le seuil de 0,8 est réglé pour la précision, pas pour le rappel.

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
Acronym correction: 6 correction(s) found, 6 applied
Acronym correction for task demo-after: 6 correction(s), 6 applied
Speaker resolution for task demo-after: source=cues (no usable metadata), 4 assigned, 0 unassigned
Acronym correction: 6 correction(s) found, 6 applied
Acronym correction for task demo-after: 6 correction(s), 6 applied
Speaker resolution for task demo-refusal: source=cues (no usable metadata), 0 assigned, 3 unassigned
```
