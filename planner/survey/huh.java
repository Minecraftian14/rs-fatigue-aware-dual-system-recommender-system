var format = """
---
link: %s
tags:
---
""";
var data = "";
data = """
1. [2510.27157] A Survey on Generative Recommendation: Data, Model, and Tasks - arXiv, accessed on February 11, 2026, 
https://arxiv.org/abs/2510.27157
2. ThinkRec: Thinking-based Recommendation via LLM - arXiv, accessed on February 11, 2026, 
https://arxiv.org/html/2505.15091v4
3. A Survey on Generative Recommendation: Data, Model, and Tasks - arXiv, accessed on February 11, 2026, 
https://arxiv.org/html/2510.27157v1
4. The 2nd Workshop on Human-Centered Recommender Systems - arXiv, accessed on February 11, 2026, 
https://arxiv.org/html/2511.19979v1
5. Enhancing deep learning recommendation systems: The transformative role of generative AI for personalized user experiences, accessed on February 11, 2026, 
https://journalwjaets.com/sites/default/files/fulltext_pdf/WJAETS-2025-0434.pdf
6. A Comprehensive Survey on LLM-Powered Recommender Systems: From Discriminative, Generative to Multi-Modal Paradigms - IEEE Xplore, accessed on February 11, 2026, 
https://ieeexplore.ieee.org/document/11129085/
7. (PDF) LLM4Rec: Large Language Models for Multimodal Generative ..., accessed on February 11, 2026, 
https://www.researchgate.net/publication/396143102_LLM4Rec_Large_Language _Models_for_Multimodal_Generative_Recommendation_with_Causal_Debiasing
8. A Survey on Large Language Models in Multimodal Recommender Systems - arXiv, accessed on February 11, 2026, 
https://arxiv.org/abs/2505.09777
9. Diffusion Models in Recommendation Systems: A Survey - arXiv, accessed on February 11, 2026, 
https://arxiv.org/html/2501.10548v1
10. SOFAI-LM: A Cognitive Architecture for Building Efficient and ..., accessed on February 11, 2026, 
https://research.ibm.com/publications/sofai-lm-a-cognitive-architecture-for-buil ding-efficient-and-reliable-reasoning-systems-with-llms
11. Second Workshop on Generative AI for Recommender Systems and Personalization (2025), accessed on February 11, 2026, 
https://genai-personalization.github.io/GenAIRecP2025
12. RecSys 2025 - Accepted Contributions, accessed on February 11, 2026, 
https://recsys.acm.org/recsys25/accepted-contributions/
13. The 2nd Workshop on Human-Centered Recommender Systems - arXiv, accessed on February 11, 2026, 
https://arxiv.org/pdf/2511.19979
14. RecSys 2024 - Accepted Contributions - ACM, accessed on February 11, 2026, 
https://recsys.acm.org/recsys24/accepted-contributions/
15. A Survey of Real-World Recommender Systems: Challenges ... - arXiv, accessed on February 11, 2026, 
https://arxiv.org/abs/2509.06002
16. A Survey of Real-World Recommender Systems: Challenges, Constraints, and Industrial Perspectives - arXiv, accessed on February 11, 2026, 
https://arxiv.org/html/2509.06002v1
17. LLM-Enhanced Reinforcement Learning for Long-Term User Satisfaction in Interactive Recommendation - arXiv, accessed on February 11, 2026, 
https://arxiv.org/html/2601.19585v2
18. Deep Reinforcement Learning for Recommender Systems | Shaped Blog, accessed on February 11, 2026, 
https://www.shaped.ai/blog/deep-reinforcement-learning-for-recommender-sys tems--a-survey
19. 8 Conversational AI Trends in 2025 - Daffodil Software, accessed on February 11, 2026, 
https://insights.daffodilsw.com/blog/8-conversational-ai-trends-in-2025
20. Conversational AI Trends for 2025: What You Need to Know - ORXTRA: Generative AI Platform for Business Growth - DXwand, accessed on February 11, 2026, 
https://dxwand.com/conversational-ai-trends/
21. The Rise of Emotional AI: How Next-Gen Chatbots Are Revolutionising Human-Machine Interaction in 2025 | by Kshitiz Sikriwal | Medium, accessed on February 11, 2026, 
https://medium.com/@beyondtheboard/the-rise-of-emotional-ai-how-next-genchatbots-are-revolutionising-human-machine-interaction-in-05eeae5c993f
22. [2511.18548] Emotion-Aware Conversational Recommender Systems: a Case Study - arXiv, accessed on February 11, 2026, 
https://www.arxiv.org/abs/2511.18548
23. A Privacy-Preserving Semi-Decentralized Personalized Recommendation System | Request PDF - ResearchGate, accessed on February 11, 2026, 
https://www.researchgate.net/publication/377602654_A_Privacy-Preserving_Semi -Decentralized_Personalized_Recommendation_System
24. DeFedGCN: Privacy-Preserving Decentralized Federated GCN for Recommender System, accessed on February 11, 2026, 
https://www.researchgate.net/publication/388545080_DeFedGCN_Privacy-Prese rving_Decentralized_Federated_GCN_for_Recommender_System
25. DeFedGCN: Privacy-Preserving Decentralized Federated GCN for Recommender System, accessed on February 11, 2026, 
https://www.computer.org/csdl/journal/sc/2025/02/10858404/23VPvI6r8eQ
26. DeFedGCN: Privacy-Preserving Decentralized Federated GCN for Recommender System, accessed on February 11, 2026, 
https://ieeexplore.ieee.org/document/10858404/
27. Position: The Current AI Conference Model is Unsustainable! Diagnosing the Crisis of Centralized AI Conferences - arXiv, accessed on February 11, 2026, 
https://arxiv.org/html/2508.04586v4""";
var lines = data.split("[\\n\\r]+");
for (int i = 0; i < lines.length; i+=2) {
}
for (int i = 0; i < lines.length; i+=2) {
    System.out.println(lines[0]);
}
for (int i = 0; i < lines.length; i+=2) {
    System.out.println(lines[i]);
}
for (int i = 0; i < lines.length; i+=2) {
    System.out.println(lines[i].replaceAll("^\\d+..", ""));
}
for (int i = 0; i < lines.length; i+=2) {
    System.out.println(lines[i].replaceAll("^\\d+..(\\[.*?\\])?", ""));
}
for (int i = 0; i < lines.length; i+=2) {
    System.out.println(lines[i].replaceAll("^\\d+..(\\[.*?\\] )?", ""));
}
for (int i = 0; i < lines.length; i+=2) {
    System.out.println(lines[i].replaceAll("^\\d+..(\\[.*?\\] )?", "").replaceAll("[^\\w -,.]", ""));
}
for (int i = 0; i < lines.length; i+=2) {
    var line = lines[i].replaceAll("^\\d+..(\\[.*?\\] )?", "").replaceAll("[^\\w -,.]", "");
    System.out.println(line.substring(0, line.lastIndexOf('-') - 1));
}
for (int i = 0; i < lines.length; i+=2) {
    var line = lines[i].replaceAll("^\\d+..(\\[.*?\\] )?", "").replaceAll("[^\\w -,.]", "");
    if (line.contains("-")) line = line.substring(0, line.lastIndexOf('-') - 1);
}
for (int i = 0; i < lines.length; i+=2) {
    var line = lines[i].replaceAll("^\\d+..(\\[.*?\\] )?", "").replaceAll("[^\\w -,.]", "");
    if (line.contains("-")) line = line.substring(0, line.lastIndexOf('-') - 1);
    System.out.println(line);
}
for (int i = 0; i < lines.length; i+=2) {
    var line = lines[i].replaceAll("^\\d+..(\\[.*?\\] )?", "").replaceAll("[^\\w -,.]", "");
    // if (line.contains("-")) line = line.substring(0, line.lastIndexOf('-') - 1);
    System.out.println(line);
}
for (int i = 0; i < lines.length; i+=2) {
    var line = lines[i].replaceAll("^\\d+..(\\[.*?\\] )?", "").replaceAll("[^\\w ,.-]", "");
    // if (line.contains("-")) line = line.substring(0, line.lastIndexOf('-') - 1);
    System.out.println(line);
}
for (int i = 0; i < lines.length; i+=2) {
    var line = lines[i].replaceAll("^\\d+..(\\[.*?\\] )?", "").replaceAll("[^\\w ,.-]", "");
    if (line.contains("-")) line = line.substring(0, line.lastIndexOf('-') - 1);
    System.out.println(line);
}
for (int i = 0; i < lines.length; i+=2) {
    var line = lines[i].replaceAll("^\\d+..(\\[.*?\\] )?", "").replaceAll("[^\\w ,.-]", "");
    // if (line.contains("-")) line = line.substring(0, line.lastIndexOf('-') - 1);
    line = line.replaceAll("( - arXiv)?, accessed on.*", "");
    System.out.println(line);
}
HashMap<String, Integer> map = new HashMap<>();
for (int i = 0; i < lines.length; i+=2) {
    var line = lines[i].replaceAll("^\\d+..(\\[.*?\\] )?", "").replaceAll("[^\\w ,.-]", "");
    // if (line.contains("-")) line = line.substring(0, line.lastIndexOf('-') - 1);
    line = line.replaceAll("( - arXiv)?, accessed on.*", "");
    if (map.containsKey(line)) line += map.get(line);
    map.put(line, map.getOrDefault(line, 0) + 1);
    System.out.println(line);
}
for (int i = 0; i < lines.length; i+=2) {
    var line = lines[i].replaceAll("^\\d+..(\\[.*?\\] )?", "").replaceAll("[^\\w ,.-]", "");
    // if (line.contains("-")) line = line.substring(0, line.lastIndexOf('-') - 1);
    line = line.replaceAll("( - arXiv)?, accessed on.*", "");
    map.put(line, map.getOrDefault(line, 0) + 1);
    System.out.println(line);
}
for (int i = 0; i < lines.length; i+=2) {
    var line = lines[i].replaceAll("^\\d+..(\\[.*?\\] )?", "").replaceAll("[^\\w ,.-]", "");
    // if (line.contains("-")) line = line.substring(0, line.lastIndexOf('-') - 1);
    line = line.replaceAll("( - arXiv)?, accessed on.*", "");
    map.put(line, map.getOrDefault(line, 0) + 1);
    if (map.containsKey(line)) line += map.get(line);
    System.out.println(line);
}
{
    for (int i = 0; i < lines.length; i+=2) {
    var line = lines[i].replaceAll("^\\d+..(\\[.*?\\] )?", "").replaceAll("[^\\w ,.-]", "");
    // if (line.contains("-")) line = line.substring(0, line.lastIndexOf('-') - 1);
    line = line.replaceAll("( - arXiv)?, accessed on.*", "");
    map.put(line, map.getOrDefault(line, 0) + 1);
    if (map.containsKey(line)) line += map.get(line);
    System.out.println(line);
}}
{
    map.clear();
    for (int i = 0; i < lines.length; i+=2) {
        var line = lines[i].replaceAll("^\\d+..(\\[.*?\\] )?", "").replaceAll("[^\\w ,.-]", "");
        // if (line.contains("-")) line = line.substring(0, line.lastIndexOf('-') - 1);
        line = line.replaceAll("( - arXiv)?, accessed on.*", "");
        map.put(line, map.getOrDefault(line, 0) + 1);
        if (map.containsKey(line)) line += map.get(line);
        System.out.println(line);
        Files.writeString(Path.of("gemres - sg", line + "md"), format.formatted(lines[i + 1]));
    }
}
{
    map.clear();
    for (int i = 0; i < lines.length; i+=2) {
        var line = lines[i].replaceAll("^\\d+..(\\[.*?\\] )?", "").replaceAll("[^\\w ,.-]", "");
        // if (line.contains("-")) line = line.substring(0, line.lastIndexOf('-') - 1);
        line = line.replaceAll("( - arXiv)?, accessed on.*", "");
        map.put(line, map.getOrDefault(line, 0) + 1);
        if (map.containsKey(line)) line += map.get(line);
        System.out.println(line);
        Files.writeString(Path.of("gemres - sg", line + ".md"), format.formatted(lines[i + 1]));
    }
}
{
    map.clear();
    for (int i = 0; i < lines.length; i+=2) {
        var line = lines[i].replaceAll("^\\d+..(\\[.*?\\] )?", "").replaceAll("[^\\w ,.-]", "");
        // if (line.contains("-")) line = line.substring(0, line.lastIndexOf('-') - 1);
        line = line.replaceAll("( - arXiv)?, accessed on.*", "");
        boolean key = map.containsKey(line);
        map.put(line, map.getOrDefault(line, 0) + 1);
        if (key) line += map.get(line);
        System.out.println(line);
        Files.writeString(Path.of("gemres - sg", line + ".md"), format.formatted(lines[i + 1]));
    }
}
