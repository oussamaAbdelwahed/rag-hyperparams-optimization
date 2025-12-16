# Détails des User stories 1

## 1 US- Flux entrant : Mapping des “Données de la commande“

Lors de la réception d’une demande de création, d’annulation de commande ou de devis, les données transmises par WOOP sont structurées selon son propre modèle de données. Le middleware effectue alors un mapping pour transformer ces données au format attendu par le modèle de données de la Standard API.

Le modèle de données de la Standard API, décrit en détail dans la section 3.2.1.3 - Modèle de données de la commande “Standard API”, est mappé avec les champs provenant des requêtes WOOP. Cette transformation garantit la conformité des commandes sur le Middleware XL EDS.

Le tableau ci-dessous présente en détail le mapping des différents champs entre le modèle WOOP et celui de la Standard API :

| **Attribut Standard API** |                   | **Description**                                                                            | **Attribut WOOP**  |            |              |
| ------------------------- | ----------------- | ------------------------------------------------------------------------------------------ | ------------------ | ---------- | ------------ |
| Référence de la commande  |                   | Identifiant unique interne Middleware                                                      | deliveryId         |            |              |
| quoteId                   |                   |                                                                                            | quoteId            |            |              |
| Type                      |                   | Interprété depuis le champ “Service”: - Type = Pickup s’il y a le service “SERVICE_RETURN” |                    |            |              |
| Prestation                |                   | Interprétée depuis le champ interval                                                       |                    |            |              |
| Client reference          |                   | Référence interne chez le client                                                           | referenceNumber    |            |              |
| Canal                     |                   | interprété depuis le champ “value” pour : “tags / Key” = “pricing”                         | tags               | value      |              |
| Items                     | reference         |                                                                                            | packages           | references | reference    |
| Items                     | name              |                                                                                            |                    |            |              |
| Items                     | type              |                                                                                            |                    |            |              |
| Items                     | description       |                                                                                            |                    |            |              |
| Items                     | quantity          |                                                                                            | packages           | quantity   |              |
| Items                     | width             |                                                                                            | packages           | width      | value        |
| Items                     | length            |                                                                                            | packages           | lenght     | value        |
| Items                     | height            |                                                                                            | packages           | height     | value        |
| Items                     | price             |                                                                                            |                    |            |              |
| Items                     | services          | Applicables à tous les articles de la commande                                             | services           |            |              |
| Service                   |                   |                                                                                            | services           |            |              |
| addresses                 | address           |                                                                                            | delivery / picking | location   | addressLine1 |
| addresses                 | address           |                                                                                            | delivery / picking | location   | addressLine2 |
| addresses                 | address           |                                                                                            | delivery / picking | location   | country      |
| addresses                 | address           |                                                                                            | delivery / picking | location   | city         |
| addresses                 | address           |                                                                                            | delivery / picking | location   | district     |
| addresses                 | address           |                                                                                            | delivery / picking | location   | postalCode   |
| addresses                 | additionalAddress |                                                                                            | delivery / picking | location   | doorCode     |
| addresses                 | additionalAddress |                                                                                            | delivery / picking | **infos**  |              |
| addresses                 | Elevator          |                                                                                            | delivery / picking | location   | elevator     |
| addresses                 | Floor             |                                                                                            | delivery / picking | location   | floor        |
| addresses                 | lat               |                                                                                            | delivery / picking | location   | coordinates  |
| addresses                 | lng               |                                                                                            | delivery / picking | location   | coordinates  |
| timeWindow                | start             |                                                                                            | delivery / picking | interval   | start        |
| timeWindow                | end               |                                                                                            | delivery / picking | interval   | end          |
| Magasin                   |                   |                                                                                            | retailer           | code       |              |
| consommateur              | Prénom            |                                                                                            | delivery / picking | location   | firstName    |
| consommateur              | Nom               |                                                                                            | delivery / picking | location   | lastName     |
| consommateur final        | Email             |                                                                                            | delivery / picking | location   | email        |
| consommateur              | N° de téléphone   |                                                                                            | delivery / picking | location   | phone        |

# Détails des User stories 2

## US “Commande: Livraison de marchandise”

| **Statut (Standard API)** |                                    |                 | **Statut (CENTIRO)**     |     |
| ------------------------- | ---------------------------------- | --------------- | ------------------------ | --- |
| **Id Statut**             | **Statut**                         | **Code Statut** | **Statut**               |
| **1**                     | Créée                              |                 |                          |
| **2**                     | Annoncée                           |                 |                          |
| **3**                     | Planning validé                    |                 |                          |
| **4**                     | Echec de chargement                | 93              | Failed delivery          |
| **5**                     | Chargement validé                  | 19              | Handed over to TSP       |
| **6**                     | En cours de livraison              | 92              | Loaded on delivery truck |
| **7**                     | Livrée                             | 99              | Order delivered          |
| 90:1                      | Delivered at PUP                   |
| **8**                     | Echec de livraison                 | 96              | Customer refusal         |
| 93                        | Failed delivery                    |
| **9**                     | Annulation XL EDS                  | 93              | Failed delivery          |
| **10**                    | Annulation Client                  |                 |                          |
| **11**                    | Annulation Client tardive          |                 |                          |
| **12**                    | Annulation Client après chargement | 98:2            | Returned to store        |
