import random
import time

from locust import HttpUser, between, task


class ChatUser(HttpUser):
    wait_time = between(5, 20)

    @task
    def ask_question(self):
        self.client.get("/")
        time.sleep(5)
        self.client.post(
            "/chat",
            json={
                "history": [
                    {
                        "user": random.choice(
                            [
                               "What is standard language for Force Majeure in a Wholesale PPA?",
                                "Who the buyer in the School House PPA?",
                                "What is the energy payment Rate in the School House PPA?",
                                "What is the end date of the contract?",

                            ]
                        )
                    }
                ],
                "approach": "rrr",
                "overrides": {
                    "retrieval_mode": "hybrid",
                    "semantic_ranker": True,
                    "semantic_captions": False,
                    "top": 3,
                    "suggest_followup_questions": False,
                },
            },
        )
        time.sleep(5)
        self.client.post(
            "/chat",
            json={
                "history": [
                    {
                        "user": "What is standard language for Force Majeure in a Wholesale PPA?",
                        "bot": "“Force Majeure” means any event that wholly or partly prevents or delays the performance by the Party affected of any obligation arising hereunder, but only if and to the extent: (a) such event or condition is not reasonably foreseeable and is not within the reasonable control of the Party affected; (b) that despite the exercise of reasonable diligence, cannot be or be caused to be prevented or avoided by such Party; and (c) such event is not the direct or indirect result of the affected Party’s negligence or the failure of such Party to perform any of its obligations under this Agreement.  Events and conditions that may constitute Force Majeure include, to the extent satisfying the foregoing requirements, events and conditions that are within or similar to one or more of the following categories: condemnation; expropriation; invasion; plague; drought; landslide; hurricane or tropical storm; tornado; lightning, ice storm, dust storm, tsunami; flood; earthquake; fire; explosion; epidemic; pandemic; quarantine; war (declared or undeclared), terrorism or other armed conflict; strikes and other labor disputes if such strike or other labor dispute is not specifically directed at the affected Party; riot or similar civil disturbance or commotion; other acts of God; acts of the public enemy; blockade; insurrection, riot or revolution; sabotage or vandalism; embargoes; and failures or delays of the Transmission System Owner directly impacting the Facility or the Interconnection Facilities but only to the extent such failures or delays were due to circumstances would constitute a “Force Majeure” impacting the Transmission System Owner.  The term “Force Majeure” shall not include (A) a Party’s ability to enter into a contract for the hedge of energy and/or sale or purchase of the Products at a more favorable price or under more favorable conditions or other economic reasons, (B) delays or nonperformance by suppliers, vendors, or other third parties with whom a Party has contracted (including interconnection or permitting delays) except to the extent that such delays or nonperformance were due to circumstances that would constitute Force Majeure, (C) serial defects of the Facility’s equipment, (D) any other economic hardship or changes in market conditions affecting the economics of either Party, (E) any delay in providing, or cancellation of, any approvals or permits by the issuing governmental authority, except to the extent such delay or cancellation was due to circumstances that would constitute Force Majeure, (F) weather conditions (including severe and extreme weather, other than the weather events expressly provided above), (G) failure or breakdown of mechanical equipment except to the extent that such delays or nonperformance were due to circumstances that would constitute Force Majeure, (H) variability in solar irradiance, including periods of low solar irradiance resulting from cloud cover, pollution, dust, smoke, weather conditions, and other causes, in the area in which the Facility is located, or (I) impacts of the COVID-19 pandemic or any mutations thereof except to the extent that, notwithstanding clause (a) above, the affected Party was not aware of, and would not reasonably be expected to have been aware of, such impacts based on information available to the affected Party as of the Effective Date. [PJM Standard Form Solar PPA (Project-Specific RECs) 2023].",

                    },
                    {"user": "Compare the definition of Force Majeure in the Standard Wholesale PPA to the School house PPA"},
                ],
                "approach": "rrr",
                "overrides": {
                    "retrieval_mode": "hybrid",
                    "semantic_ranker": True,
                    "semantic_captions": False,
                    "top": 3,
                    "suggest_followup_questions": False,
                },
            },
        )
