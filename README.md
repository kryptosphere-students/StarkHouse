# StarkHouse 🏡

![Banner](/assets/Banner.png)

## Short Description

StarkHouse is a hackathon project at MuBuenos in Argentina in April 2024. 

Starkhouse leverages Giza tech, a ZKML database to provide fair and secure housing matching between potential renters and house providers on the Argentinian market. This project aims to leverage efficient and fair matching on a secure database, not accessible to any third party. This app enables the treatment of highly sensitive data. The model can be extended to any matching system on any population.

## Problem

### Inefficiency housing attribution

Currently, we can remark an inefficiency in housing attribution, with an unfairness of the system not based neither on the urgency nor the need.

### Lack of privacy

Currently, when you're looking for to rent an apartment, you need to provide some sensitive information like the amount on your bank account, for instance.

## How does it built

StarkHouse MVP leveraged ZKML, which is powered by a custom machine-learning system based on an optimized linear regression. Trained on a curated Dataset of rental properties in Argentina and matched with a dataset of Argentinian credentials and revenues.

- [Giza AI actions](https://actions.gizatech.xyz/) to 
- [Orion](https://orion.gizatech.xyz/) (Giza) to generate proofs and verify them
- [Linear Regression](https://actions.gizatech.xyz/tutorials/traditional-ml-models-for-zkml/linear-regression) to establish a score and match houses and users
- [QuarkID](https://quarkid.org/) to use an Argentinian ZK database
- Database of rental properties in Argentina

### Steps

1 - Access to a private database (QuarkID / Rental properties)


2 - Use the algorithm to have a ranking of 5 more promising houses using linear regression for scoring per person with an extensive filter

![Matrix](/assets/Matrix.png)


3 - Verify the ranking is fair with verifiable Giza proof

![Proof](/assets/proof.png)

## Get started with our documentation

Coming soon.

## How does it work

StarkHouse is a verified system providing proof of fair matching on Starknet provided by Giza. This application can power any private or public housing marketplace with fair and efficient matching.

### Housing scoring

![Housing Scoring](/assets/Housing-scoring.png)

### Guarantee privacy of sensitive data

User going to encrypt their sensitive data to rent a house as verifiable credentials.

## Roadmap

- New use cases unblocked using any matching system with sensitive data (grants, marketplace, rewards)
- SDK to improve the matching system on any housing marketplace
- Leverage Herodotus collaboration with Giza to ensure data authenticity and provenance

### About the SDK

- Public systems: any allocation system that matches people with needs
- Private systems: custom matching programs depending of the need

## Contributors

- Jules Foa [@julesfoa](https://github.com/julesfoa)
- Nandy Bâ: [@nandyba](https://github.com/nandyba)
- Solène Daviaud: [@sdaav](https://github.com/sdaav)

