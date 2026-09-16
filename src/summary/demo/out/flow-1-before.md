

 **SPEAKER_00**:  Bonjour à toutes et à tous.  Ici Nadia. On ouvre le point d'architecture hebdomadaire sur La Suite.  Trois sujets à l'ordre du jour : la collaboration en temps réel dans Docs, le stockage objet, et l'intégration des mini-applications.  On a quarante minutes, donc on garde les débats techniques pour la fin.  Julien, tu démarres ?

 **SPEAKER_01**:  Bonjour, Julien à l'appareil. Côté dix nomme, la brique de collaboration a bien avancé ce mois-ci.  Le serveur temps réel tourne maintenant entièrement sur y grec js.  Le modèle de données repose sur des serre des thés, donc la fusion des modifications se fait sans verrou côté serveur.  On a retiré le dernier verrou pessimiste la semaine dernière.

 **SPEAKER_00**:  Et la reconnexion après une coupure réseau, c'est réglé ?

 **SPEAKER_01**:  Pas encore. Au-delà de treize secondes hors ligne, le client renvoie l'état complet du document au lieu d'un delta.  Sur un document long, ça provoque un pic de mémoire côté back-end, et parfois une fusion qui duplique un paragraphe.

 **SPEAKER_02**:  Mathilde, bonjour. Est-ce que ça se voit côté utilisateur ?

 **SPEAKER_01**:  Oui, une latence de deux à trois secondes à la reprise.  Ce n'est pas bloquant, mais c'est visible, surtout sur les documents partagés à plus de dix personnes.

 **SPEAKER_00**:  On le garde en risque ouvert pour l'instant.  Karim, le stockage ?

 **SPEAKER_03**:  Karim, bonjour à tous. Les deux applications écrivent dans mini eau, mais avec deux conventions de nommage différentes.  Drive préfixe les objets par identifiant d'espace, Docs par identifiant de document.  Pour la sauvegarde et pour la purge, ça complique beaucoup le tri.  On ne sait pas dire, aujourd'hui, quels objets appartiennent à quel service.

 **SPEAKER_02**:  Et pour Drive, on passe toujours par whoopee pour l'édition bureautique ?

 **SPEAKER_03**:  Oui. Le protocole impose son propre système de jetons, donc on se retrouve avec deux chaînes d'autorisation en parallèle.  C'est le principal point de dette technique sur cette partie.  Tant qu'on ne l'a pas résorbé, chaque évolution d'authentification doit être faite deux fois.

 **SPEAKER_00**:  Julien, où en est l'unification de l'authentification ?

 **SPEAKER_01**:  En local, on garde qui cloaque pour le développement.  En production, c'est ProConnect. Le flux oh idée sait est identique, seule l'URL de découverte change.  Les équipes qui arrivent sur le projet se trompent souvent là-dessus, donc il faut une note claire.

 **SPEAKER_02**:  Et l'échange de jetons entre applications, on a tranché ?

 **SPEAKER_01**:  On passe par main chêne, qui implémente l'échange de jetons décrit dans la RFC 8693.  Ça permet à Docs d'appeler l'API de Drive au nom de l'utilisateur, sans redemander une authentification.  C'est encore expérimental, mais la mécanique fonctionne en intégration.

 **SPEAKER_03**:  Attention quand même : ce composant n'est pas encore empaqueté pour notre chaîne de déploiement.  Il faudra prévoir un vrai travail d'industrialisation, avec les sondes et les journaux qui vont avec.

 **SPEAKER_00**:  Noté. Mathilde, les mini-applications.

 **SPEAKER_02**:  On a dix-huit demandes remontées par les ministères.  La plus fréquente, de loin, c'est l'export en PDF depuis Docs.  Le prototype utilise type est avec un gabarit qui reprend la charte Marie-Anne et le bloc marque de l'État.

 **SPEAKER_01**:  Le problème, c'est que Docs n'expose pas d'export côté serveur.  Tout se fait dans le navigateur aujourd'hui, donc on ne peut pas générer un document depuis une tâche planifiée.

 **SPEAKER_02**:  Soit on ajoute un point d'entrée côté back-end, soit on passe par dos pecs, le micro-service de conversion de documents.

 **SPEAKER_01**:  Ce service est prévu pour convertir du DOCX vers bloc note, pas l'inverse.  Il faudrait écrire le chemin retour, et ce n'est pas une petite tâche.

 **SPEAKER_00**:  Chiffrez les deux options, on tranchera hors réunion avec le sponsor.

 **SPEAKER_02**:  D'accord. Dernier sujet : les tableurs.  Est-ce qu'on continue à stocker les fichiers Christ dans Drive, ou est-ce qu'on attend le prototype interne ?

 **SPEAKER_03**:  On les stocke, mais on ne les intègre pas.  C'est un logiciel tiers, avec sa propre base de données et sa propre authentification.  Le jour où on voudra l'intégrer vraiment, il faudra reprendre le sujet depuis le début.

 **SPEAKER_01**:  Et il n'y a toujours pas de réponse sur les widgets.  La question avait déjà été posée l'an dernier pendant les journées de développement, elle est restée sans suite.

 **SPEAKER_00**:  On récapitule les actions. Julien documente le flux entre Keycloak et ProConnect pour les nouveaux arrivants.  Karim ouvre un ticket sur le nommage des objets dans le stockage.  Mathilde chiffre les deux options d'export PDF, avec une estimation en jours-homme.

 **SPEAKER_03**:  J'ai besoin d'un accès à la pré-production pour tester la sauvegarde.  Je passe par qui ?

 **SPEAKER_00**:  Par la DSI, avec une demande sur l'outil de tickets.  Mets-moi en copie, je relancerai si besoin.

 **SPEAKER_02**:  Une dernière chose : la prochaine réunion tombe pendant les congés scolaires.  On décale d'une semaine ?

 **SPEAKER_00**:  On décale. Je renvoie une invitation cet après-midi.  Merci à toutes et à tous, bonne journée.