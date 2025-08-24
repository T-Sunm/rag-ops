# Định nghĩa các dataset với metadata
DATASETS = {
    "environment_battery": {
        "data": [
            {
                "title": "List of unit recycling products, packaging from the Ministry of Natural Resources and Environment-VN",
                "url": "https://sotnmt.soctrang.gov.vn/SiteFolders/stn/4.Nam2024/460-TB-BTNMT.pdf",
            },
            {
                "title": "Research on the responsibility of recycling batteries and rechargeable batteries-VN",
                "url": "https://epr.vn/upload/bao-cao-nghien-cuu/1681609661643b53bd1a8109.68128269.pdf",
            },
            {
                "title": "EPA Workshop Report on Lithium-ion Battery Handling and Recycling-USA",
                "url": "https://www.epa.gov/system/files/documents/2022-03/final_lithium-ion-battery-workshop-public-report_508web.pdf",
            },
            {
                "title": "US Congressional Document on Legislation and Funding to Improve Recycling Access-USA",
                "url": "https://www.congress.gov/117/meeting/house/114965/documents/HHRG-117-IF18-20220630-SD002.pdf",
            },
            {
                "title": "New York State Rechargeable Battery Law Monitoring and Enforcement Report-NY",
                "url": "https://www.osc.ny.gov/files/state-agencies/audits/pdf/sga-2022-21s19.pdf",
            },
            {
                "title": "World Economic Forum report on building a circular battery economy-INGO",
                "url": "https://reports.weforum.org/docs/WEF_Powering_the_Future_2025.pdf",
            },
            {
                "title": "WWF research on responsible battery manufacturing and EU regulations-INGO",
                "url": "https://www.systemiq.earth/wp-content/uploads/2024/04/wwf-study.pdf",
            },
            {
                "title": "Research on promoting sustainable battery recycling towards a circular battery system-INGO",
                "url": "https://www.systemiq.earth/wp-content/uploads/2023/11/Systemiq_Sustainable_Battery_Recycling_Full_Study_WEB-1.pdf",
            },
            {
                "title": "Greenpeace investigation into trade in toxic lead-acid battery waste-INGO",
                "url": "https://www.greenpeace.to/greenpeace/wp-content/uploads/2019/09/LEAD-ASTRAY-THE-POISONOUS-LEAD-BATTERY-WASTE-TRADE_GP-1994.pdf",
            },
        ],
        "description": "Environmental and battery recycling documents",
    },
    "llm_papers": {
        "data": [
            {
                "title": "Attention Is All You Need",
                "url": "https://arxiv.org/pdf/1706.03762",
            },
            {
                "title": "BERT- Pre-training of Deep Bidirectional Transformers for Language Understanding",
                "url": "https://arxiv.org/pdf/1810.04805",
            },
            {
                "title": "Chain-of-Thought Prompting Elicits Reasoning in Large Language Models",
                "url": "https://arxiv.org/pdf/2201.11903",
            },
            {
                "title": "Denoising Diffusion Probabilistic Models",
                "url": "https://arxiv.org/pdf/2006.11239",
            },
            {
                "title": "Instruction Tuning for Large Language Models- A Survey",
                "url": "https://arxiv.org/pdf/2308.10792",
            },
            {
                "title": "Llama 2- Open Foundation and Fine-Tuned Chat Models",
                "url": "https://arxiv.org/pdf/2307.09288",
            },
        ],
        "description": "AI/ML research papers",
    },
}

# Backward compatibility
environment_battery = DATASETS["environment_battery"]["data"]
llm_papers = DATASETS["llm_papers"]["data"]


# Dynamic function to get all dataset names
def get_dataset_names():
    return list(DATASETS.keys())


def get_dataset_by_name(name):
    return DATASETS.get(name, {}).get("data", [])
