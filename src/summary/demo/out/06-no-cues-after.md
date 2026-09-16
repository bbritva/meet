

 **SPEAKER_00**:  Bonjour à tous. On enchaîne sur le point hebdomadaire infrastructure.  Deux sujets ce matin : l'incident de jeudi dernier sur la supervision, et la préparation de la fenêtre de maintenance.  On commence par l'incident, comme Damien l'avait signalé au dernier comité.

 **SPEAKER_01**:  L'incident a duré une heure quarante.  La cause immédiate est un disque saturé sur le collecteur de métriques.  La cause profonde, c'est qu'aucune alerte ne surveille le remplissage de ce disque en particulier, alors que toutes les autres partitions sont  couvertes.

 **SPEAKER_00**:  Pourquoi cette partition est-elle passée à travers ?

 **SPEAKER_01**:  Elle a été ajoutée après la mise en place de la supervision, lors d'une extension de capacité.  Le modèle d'alerte n'a pas été régénéré, et personne ne l'a vu parce que le tableau de bord n'affiche que les partitions  déjà connues.

 **SPEAKER_02**:  C'est un schéma qu'on a déjà rencontré sur le stockage objet l'an dernier.  La supervision décrit l'infrastructure telle qu'elle était au moment de son installation, pas telle qu'elle est aujourd'hui.  Tant qu'elle n'est pas générée à partir de l'inventaire réel, ça se reproduira.

 **SPEAKER_00**:  Est-ce qu'on a un inventaire réel exploitable ?

 **SPEAKER_02**:  Partiellement. L'inventaire existe pour les machines virtuelles, pas pour les volumes attachés.  Il faudrait étendre la collecte, ce qui représente une charge de deux à trois semaines, et une modification du droit de lecture  sur l'hyperviseur.

 **SPEAKER_01**:  La modification du droit de lecture demande un passage en comité de sécurité, parce que le référentiel de l'ANSSI l'impose.  Ce n'est pas bloquant, mais il faut compter un mois de délai, parce que le comité ne siège qu'une fois par mois.

 **SPEAKER_00**:  On lance la demande cette semaine alors, sinon nous serons en février.  Sujet suivant : la fenêtre de maintenance.

 **SPEAKER_02**:  Elle est prévue le premier week-end d'octobre.  Trois opérations sont prévues : la montée de version du serveur de bases, le remplacement d'un commutateur, et la rotation des certificats  internes. Le PCA a été relu la semaine dernière.  Les trois sont indépendantes, mais elles partagent la même fenêtre.

 **SPEAKER_01**:  Trois opérations sur une seule fenêtre, c'est beaucoup.  Si la montée de version dérape, nous n'aurons pas le temps de revenir en arrière avant l'ouverture du lundi matin.  Je propose de sortir la rotation des certificats et de la traiter la semaine suivante.

 **SPEAKER_00**:  Quel est le risque à décaler la rotation ?

 **SPEAKER_02**:  Faible : le certificat le plus proche expire à la mi-novembre.  Une semaine de décalage ne change rien, et ça allège franchement la fenêtre.  Je suis favorable à la proposition.

 **SPEAKER_00**:  Alors on fait comme ça.  Deux opérations le premier week-end, la rotation le week-end suivant.  Je mets à jour le calendrier de maintenance, je préviens les équipes applicatives et la DSI.  Bonne journée à tous.