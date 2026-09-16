

 **Nadia Berger**:  Bonjour à toutes et à tous.  Ici Nadia. On ouvre le point d'architecture hebdomadaire sur La Suite.  Trois sujets à l'ordre du jour : la collaboration en temps réel dans Docs, le stockage objet, et l'intégration des mini-applications.  On a quarante minutes, donc on garde les débats techniques pour la fin.  Julien, tu démarres ?

 **Julien Perrot**:  Bonjour, Julien à l'appareil. Côté dix nomme, la brique de collaboration a bien avancé ce mois-ci.  Le serveur temps réel tourne maintenant entièrement sur Yjs.  Le modèle de données repose sur des CRDT, donc la fusion des modifications se fait sans verrou côté serveur.  On a retiré le dernier verrou pessimiste la semaine dernière.

 **Nadia Berger**:  Et la reconnexion après une coupure réseau, c'est réglé ?

 **Julien Perrot**:  Pas encore. Au-delà de treize secondes hors ligne, le client renvoie l'état complet du document au lieu d'un delta.  Sur un document long, ça provoque un pic de mémoire côté back-end, et parfois une fusion qui duplique un paragraphe.

 **Mathilde Roux**:  Mathilde, bonjour. Est-ce que ça se voit côté utilisateur ?

 **Julien Perrot**:  Oui, une latence de deux à trois secondes à la reprise.  Ce n'est pas bloquant, mais c'est visible, surtout sur les documents partagés à plus de dix personnes.

 **Nadia Berger**:  On le garde en risque ouvert pour l'instant.  Karim, le stockage ?

 **Karim Sahli**:  Karim, bonjour à tous. Les deux applications écrivent dans MinIO, mais avec deux conventions de nommage différentes.  Drive préfixe les objets par identifiant d'espace, Docs par identifiant de document.  Pour la sauvegarde et pour la purge, ça complique beaucoup le tri.  On ne sait pas dire, aujourd'hui, quels objets appartiennent à quel service.

 **Mathilde Roux**:  Et pour Drive, on passe toujours par WOPI pour l'édition bureautique ?

 **Karim Sahli**:  Oui. Le protocole impose son propre système de jetons, donc on se retrouve avec deux chaînes d'autorisation en parallèle.  C'est le principal point de dette technique sur cette partie.  Tant qu'on ne l'a pas résorbé, chaque évolution d'authentification doit être faite deux fois.

 **Nadia Berger**:  Julien, où en est l'unification de l'authentification ?

 **Julien Perrot**:  En local, on garde Keycloak pour le développement.  En production, c'est ProConnect. Le flux OIDC est identique, seule l'URL de découverte change.  Les équipes qui arrivent sur le projet se trompent souvent là-dessus, donc il faut une note claire.

 **Mathilde Roux**:  Et l'échange de jetons entre applications, on a tranché ?

 **Julien Perrot**:  On passe par Menshen, qui implémente l'échange de jetons décrit dans la RFC 8693.  Ça permet à Docs d'appeler l'API de Drive au nom de l'utilisateur, sans redemander une authentification.  C'est encore expérimental, mais la mécanique fonctionne en intégration.

 **Karim Sahli**:  Attention quand même : ce composant n'est pas encore empaqueté pour notre chaîne de déploiement.  Il faudra prévoir un vrai travail d'industrialisation, avec les sondes et les journaux qui vont avec.

 **Nadia Berger**:  Noté. Mathilde, les mini-applications.

 **Mathilde Roux**:  On a dix-huit demandes remontées par les ministères.  La plus fréquente, de loin, c'est l'export en PDF depuis Docs.  Le prototype utilise type est avec un gabarit qui reprend la charte Marie-Anne et le bloc marque de l'État.

 **Julien Perrot**:  Le problème, c'est que Docs n'expose pas d'export côté serveur.  Tout se fait dans le navigateur aujourd'hui, donc on ne peut pas générer un document depuis une tâche planifiée.

 **Mathilde Roux**:  Soit on ajoute un point d'entrée côté back-end, soit on passe par dos pecs, le micro-service de conversion de documents.

 **Julien Perrot**:  Ce service est prévu pour convertir du DOCX vers bloc note, pas l'inverse.  Il faudrait écrire le chemin retour, et ce n'est pas une petite tâche.

 **Nadia Berger**:  Chiffrez les deux options, on tranchera hors réunion avec le sponsor.

 **Mathilde Roux**:  D'accord. Dernier sujet : les tableurs.  Est-ce qu'on continue à stocker les fichiers Christ dans Drive, ou est-ce qu'on attend le prototype interne ?

 **Karim Sahli**:  On les stocke, mais on ne les intègre pas.  C'est un logiciel tiers, avec sa propre base de données et sa propre authentification.  Le jour où on voudra l'intégrer vraiment, il faudra reprendre le sujet depuis le début.

 **Julien Perrot**:  Et il n'y a toujours pas de réponse sur les widgets.  La question avait déjà été posée l'an dernier pendant les journées de développement, elle est restée sans suite.

 **Nadia Berger**:  On récapitule les actions. Julien documente le flux entre Keycloak et ProConnect pour les nouveaux arrivants.  Karim ouvre un ticket sur le nommage des objets dans le stockage.  Mathilde chiffre les deux options d'export PDF, avec une estimation en jours-homme.

 **Karim Sahli**:  J'ai besoin d'un accès à la pré-production pour tester la sauvegarde.  Je passe par qui ?

 **Nadia Berger**:  Par la DSI, avec une demande sur l'outil de tickets.  Mets-moi en copie, je relancerai si besoin.

 **Mathilde Roux**:  Une dernière chose : la prochaine réunion tombe pendant les congés scolaires.  On décale d'une semaine ?

 **Nadia Berger**:  On décale. Je renvoie une invitation cet après-midi.  Merci à toutes et à tous, bonne journée.