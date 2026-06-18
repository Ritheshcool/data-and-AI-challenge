# Redrob Candidate Pool — Data Profile

- Total candidates: **100000**
- Recency anchor (max last_active_date): **2026-06-01**
- open_to_work: 35339 (35.3%); willing_to_relocate: 28804 (28.8%)

## Geography
- Countries (top): [('India', 75113), ('USA', 9978), ('Australia', 2579), ('Canada', 2506), ('UK', 2472), ('Germany', 2469), ('Singapore', 2453), ('UAE', 2430)]
- India cities (top): [('bhubaneswar', 4321), ('noida', 4283), ('hyderabad', 4283), ('jaipur', 4268), ('bangalore', 4238), ('kolkata', 4230), ('indore', 4198), ('pune', 4186), ('chennai', 4164), ('delhi', 4161), ('trivandrum', 4151), ('ahmedabad', 4143), ('chandigarh', 4128), ('coimbatore', 4113), ('vizag', 4093), ('kochi', 4073), ('mumbai', 4043), ('gurgaon', 4037)]

## Titles / Industry / Company
- current_title (top 30): [('Business Analyst', 5833), ('HR Manager', 5830), ('Mechanical Engineer', 5791), ('Accountant', 5764), ('Project Manager', 5754), ('Customer Support', 5750), ('Operations Manager', 5744), ('Content Writer', 5727), ('Sales Executive', 5713), ('Civil Engineer', 5702), ('Graphic Designer', 5689), ('Marketing Manager', 5524), ('Software Engineer', 3450), ('Full Stack Developer', 2873), ('Cloud Engineer', 2836), ('Java Developer', 2809), ('.NET Developer', 2788), ('DevOps Engineer', 2787), ('Mobile Developer', 2757), ('Frontend Engineer', 2738), ('QA Engineer', 2682), ('Analytics Engineer', 764), ('Data Engineer', 744), ('Data Analyst', 728), ('Backend Engineer', 704), ('Senior Data Engineer', 687), ('Senior Software Engineer', 653), ('ML Engineer', 167), ('AI Research Engineer', 153), ('Data Scientist', 145)]
- current_industry (top 20): [('IT Services', 29881), ('Software', 22417), ('Manufacturing', 22305), ('Conglomerate', 7571), ('Paper Products', 7467), ('Fintech', 2808), ('Food Delivery', 2514), ('E-commerce', 1529), ('Consulting', 1274), ('EdTech', 610), ('SaaS', 328), ('AI/ML', 278), ('AdTech', 172), ('Transportation', 162), ('Insurance Tech', 155), ('Gaming', 149), ('HealthTech', 147), ('HealthTech AI', 68), ('Conversational AI', 62), ('AI Services', 42)]
- current_company (top 30): [('Infosys', 7590), ('Wayne Enterprises', 7571), ('Wipro', 7566), ('Initech', 7528), ('Pied Piper', 7500), ('Globex Inc', 7492), ('Acme Corp', 7490), ('Dunder Mifflin', 7467), ('TCS', 7451), ('Hooli', 7378), ('Stark Industries', 7323), ('Swiggy', 1288), ('Accenture', 1274), ('Capgemini', 1265), ('CRED', 1257), ('HCL', 1250), ('Razorpay', 1246), ('Zomato', 1226), ('Mindtree', 1225), ('Cognizant', 1213), ('Flipkart', 1171), ('Tech Mahindra', 1168), ('Mphasis', 1153), ('Meesho', 186), ('InMobi', 172), ('Nykaa', 172), ('Zoho', 165), ('Freshworks', 163), ('Vedantu', 163), ('Ola', 161)]

## Distributions
- years_of_experience: {'min': 1.0, 'p10': 2.2, 'median': 6.8, 'mean': 7.17, 'p90': 13.0, 'max': 16.9}
- profile_completeness: {'min': 25.0, 'p10': 32.8, 'median': 56.8, 'mean': 56.76, 'p90': 80.4, 'max': 99.9}
- recruiter_response_rate: {'min': 0.02, 'p10': 0.14, 'median': 0.44, 'mean': 0.44, 'p90': 0.73, 'max': 0.95}
- github_activity_score (incl -1): {'min': -1, 'p10': -1, 'median': -1.0, 'mean': 9.62, 'p90': 40.4, 'max': 96.9}
- num_skills per candidate: {'min': 5, 'p10': 6, 'median': 9.0, 'mean': 9.6, 'p90': 14, 'max': 23}
- education tiers: {'tier_3': 53220, 'tier_4': 51885, 'tier_2': 27821, 'tier_1': 6852}
- preferred_work_mode: {'onsite': 25000, 'flexible': 25000, 'hybrid': 25076, 'remote': 24924}

## JD-skill area coverage
- candidates with area term in SKILLS list: {'vec_db': 12866, 'llm_nlp': 14372, 'neg_domain': 31870, 'retrieval': 12486, 'ranking': 6393}
- candidates with area term in CAREER descriptions: {'llm_nlp': 36948, 'ranking': 2330, 'eval': 1734, 'retrieval': 9023, 'neg_domain': 509, 'vec_db': 108}

## Honeypot / impossibility checks
- per-check counts: {'skill_dur_gt_career': 9231, 'active_before_signup': 7496, 'salary_min_gt_max': 18865, 'careerspan_gt_yoe': 4, 'expert_zero_duration': 21, 'duration_vs_dates_mismatch': 33}
- distribution of #flags-per-candidate: {0: 67878, 1: 28700, 2: 3316, 3: 106}
  (spec says ~80 honeypots forced to tier 0; candidates with >=2 flags are prime suspects)

## Trap-category candidate counts
- consulting-only careers: 9745 (9.7%)
- keyword stuffers (nontech title + >=4 AI skill areas): 4570 (4.6%)
- plain-language gems (career shows retrieval/recsys, <=1 buzzword skill, tech title): 8976 (9.0%)
- perfect-on-paper but inactive/unresponsive: 3772 (3.8%)
