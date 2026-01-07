<a id="readme-top"></a>

<!-- PROJECT SHIELDS -->
[![Issues][issues-shield]][issues-url]
[![project_license][license-shield]][license-url]
[![LinkedIn][linkedin-shield]][linkedin-url]

<h3 align="center">Yr 2 DAT5501 Final Project 🚀</h3>

  <p align="center">
    This repository contains the final project for the DAT5501 Analysis Software and Career Practice Module. 
    <br />
    <a href="https://github.com/phizzii/DAT5501-Final-Project"><strong>Explore the docs »</strong></a>

<p align="center">⋆｡ﾟ☁︎｡⋆｡ ﾟ☾ ﾟ｡⋆ ─── ⋆⋅☆⋅⋆ ─── ⋆｡ﾟ☁︎｡⋆｡ ﾟ☾ ﾟ｡⋆ ─── ⋆⋅☆⋅⋆ ─── ⋆｡ﾟ☁︎｡⋆｡ ﾟ☾ ﾟ｡⋆ ─── ⋆⋅☆⋅⋆ ─── ⋆｡ﾟ☁︎｡⋆｡ ﾟ☾ ﾟ｡⋆</p>

<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li><a href="#roadmap">Roadmap</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
  </ol>
</details>

<!-- ABOUT THE PROJECT -->
## About The Project

This repository will showcase an authentic business problem in which I will analyse multiple different high-quality datasets, cleaning data, creating figures to support new found insights from my analysis. Additionally, I will build a professional data analysis workflow to develop a solution. This final project has also supported my upskilling in creating testing suites for my scripts linking CircleCI so when changes are pushed, I can see whether or not my tests that I have written for each piece of code has passed or not.

For this project I have chosen to analyse user-generated review datasets from Amazon, Steam and Yelp. I also aimed to produce interpretable and reusable scripts as a data-driven business solution for companies to use as a way to detecting negative user reviews before aggregate ratings and sale potential fall.

The pipeline begins with reproducible dataset sampling and storage. Amazon category reviews are streamed and downsampled from the McAuley-Lab Amazon Reviews 2023 corpus on Hugging Face; Steam and Yelp datasets are also sourced from Hugging Face (with Steam requiring additional local preprocessing in the raw data workflow). Datasets are sampled with fixed random seeds and persisted to disk so data collection is not repeated unnecessarily. Personally identifiable information is not used, and analysis is performed at the review level using anonymised public datasets.

For modelling, each platform is processed into a consistent set of interpretable features designed to capture dissatisfaction signals beyond star ratings alone. Reviews are binarised into “negative” versus “non-negative” outcomes (with neutral reviews excluded to reduce ambiguity). Textual features are extracted from review text using sentiment polarity and subjectivity (TextBlob), alongside structural and expressive cues such as word/character length, average sentence length, exclamation frequency, and capitalisation ratio. Platform-specific metadata is incorporated where available (e.g., verified purchase, helpful votes, and review age for Amazon; playtime, helpfulness, and early access indicators for Steam). A Logistic Regression model is trained per dataset using a scikit-learn pipeline with appropriate preprocessing (one-hot encoding for categorical fields, standardisation for numeric fields, and class weighting to address imbalance). Coefficients are exported for transparency and business interpretability.

Results consistently show that sentiment polarity and verbosity are among the strongest predictors of negative experience across platforms, supporting the use of linguistic signals as an early indicator of dissatisfaction. At the same time, coefficient patterns differ meaningfully by platform and context—reflecting differences in review norms, interface design, and user expectations (e.g., the role of verified purchase on Amazon and early access on Steam). These differences reinforce the need for platform-aware monitoring rather than one-size-fits-all reputation management strategies.

The repository also includes automated visualisation scripts that generate cross-platform comparison figures (e.g., top coefficients per dataset, sentiment distributions for negative reviews, and sentiment versus outcome groups). Continuous integration is configured via CircleCI: on every push, CircleCI runs a pytest suite that tests data scripts, model scripts, and plotting scripts using lightweight mocks (to avoid large downloads and to keep CI fast). This ensures that the end-to-end workflow remains reproducible and stable.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

### Built With

[![Python][Python]][python-url]

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- ROADMAP -->
## Roadmap

- [ ] Upload everything!

See the [open issues](https://github.com/phizzii/DAT5501-Final-Project/issues) for a full list of proposed features (and known issues).

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- LICENSE -->
## License

MIT license for learning :)!

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- CONTACT -->
## Contact

Sophie - [@VolarPhizzie](https://x.com/VolarPhizzie)

Project Link: [https://github.com/phizzii/DAT5501-Final-Project](https://github.com/phizzii/DAT5501-Final-Project)

LinkedIn: [Add me here!!](https://www.linkedin.com/in/sophie-botten-82a91227a/)

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- ACKNOWLEDGMENTS -->
## Citations

- McAuley J et al, Hugging Face, Bridging Language and Items for Retrieval and Recommendation, 2024, URL: https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023
-  McAuley J et al, Hugging Face, Self-attentive sequential recommendation, 2018, URL: https://huggingface.co/datasets/recommender-system/steam-review-and-bundle-dataset
- Zhang Z, Hugging Face, Yelp Dataset Challenge, 2015, URL: https://huggingface.co/datasets/Yelp/yelp_review_full 


<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[issues-shield]: https://img.shields.io/github/issues/phizzii/DAT5501-Final-Project.svg?style=for-the-badge
[issues-url]: https://github.com/phizzii/DAT5501-Final-Project/issues
[license-shield]: https://img.shields.io/github/license/phizzii/DAT5501-Final-Project.svg?style=for-the-badge
[license-url]: https://github.com/phizzii/DAT5501-Final-Project/LICENSE.md
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-url]: https://www.linkedin.com/in/sophie-botten-82a91227a/
<!-- Shields.io badges. You can a comprehensive list with many more badges at: https://github.com/inttter/md-badges -->
[Python]: https://img.shields.io/badge/python-FF87E3?style=for-the-badge&logo=python&logoColor=white
[python-url]: https://www.python.org
