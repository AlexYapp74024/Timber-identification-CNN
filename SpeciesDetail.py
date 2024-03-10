import re

class SpeciesDetail():
    def __init__(self, name, desc, link, detail) -> None:
        self.name = name
        self.desc = desc
        self.link = link
        self.detail = detail

    def result_text(self):
        detail:str = self.detail
        detail = re.sub('#####(.+)',r'<h5>\g<1></h5>',detail)
        detail = re.sub('####(.+)',r'<h4>\g<1></h4>',detail)
        detail = re.sub('###(.+)',r'<h3>\g<1></h3>',detail)
        return f'''## Prediction: {self.name}
<details>
<summary>Details</summary>
{detail}
</details>
'''

labels = [
    SpeciesDetail('Araucaria angustifolia', '''
Araucaria angustifolia, the Paraná pine, Brazilian pine or candelabra tree (pinheiro-do-paraná, araucária or pinheiro brasileiro), is a critically endangered species in the conifer genus Araucaria. Although the common names in various languages refer to the species as a "pine", it does not belong in the genus Pinus. 
''', "https://en.wikipedia.org/wiki/Araucaria_angustifolia", '''
Araucaria angustifolia, the Paraná pine, Brazilian pine or candelabra tree (pinheiro-do-paraná, araucária or pinheiro brasileiro), is a critically endangered species in the conifer genus Araucaria. Although the common names in various languages refer to the species as a "pine", it does not belong in the genus Pinus.
###Origin and taxonomy

The genus Araucaria was part of terrestrial flora since the Triassic and found its apogee in Gondwana. Today, it is restricted to the Southern Hemisphere and has 19 species.
###Distribution

Covering an original area of 233,000 square kilometres (90,000 sq mi), it has now lost an estimated 97% of its habitat to logging, agriculture, and silviculture.

It is native to southern Brazil (also found in high-altitude areas of southern Minas Gerais, in central Rio de Janeiro and in the east and south of São Paulo, but more typically in the states of Paraná, Santa Catarina and Rio Grande do Sul). According to a study made by Brazilian researcher Reinhard Maack, the original area of occurrence represented 36.67% of the Paraná state (73,088 km2 or 28,219 sq mi), 60.13% of the Santa Catarina state (57,332 km2 or 22,136 sq mi), 21.6% of the São Paulo state (53,613 km2 or 20,700 sq mi) and 17.38% of the Rio Grande do Sul state (48,968 km2 or 18,907 sq mi). It is also found in the northeast of Argentina (Misiones and Corrientes), locally in Paraguay (Alto Paraná), growing in low mountains at altitudes of 500–1,800 metres (1,600–5,900 ft) and in northern regions of Uruguay where it was thought to be extinct until recent discoveries.

The prehistoric distribution of A. angustifolia in earlier geologic periods was very different to the present day, fossils were found in northeastern Brazil. The present-day range is recent, the species moving into this area during the later Pleistocene and early Holocene. This chorological shift may possibly be due to climatic change and the migration of mountain flora by way of river courses.
###Description

It is an evergreen tree growing to 40 m (130 ft) tall and 1 m (3 ft 3 in) diameter at breast height. However, the largest individual, near Nova Petrópolis, Rio Grande do Sul state, Brazil is 147.7 feet (45 m) in height with a D.B.H. (diameter at breast height) of 12.5 feet (3.8 m) girth. The tree is fast growing; as much as 3 feet 8 inches (113 cm) a year (16 metres (52 ft) in 14 years) at Puerto Piray, Misiones Province, Argentina.: 13_8  The leaves are thick, tough and scale like, triangular, 3–6 centimetres (1+1⁄8–2+3⁄8 in) long, 5–10 millimetres (25⁄128–25⁄64 in) broad at the base, and with razor-sharp edges and tip. They persist 10 to 15 years, so cover most of the tree except for the trunk and older branches. The bark is uncommonly thick; up to six inches (15 centimeters) deep. It is closely related to Araucaria araucana from further southwest in South America, differing most conspicuously in the narrower leaves.

It is usually dioecious, with the male and female cones on separate trees. The male (pollen) cones are oblong, 6 cm (2+1⁄2 in) long at first, expanding to 10–18 cm (4–7 in) long by 15–25 mm (19⁄32–63⁄64 in) broad at pollen release. Like all conifers it is wind pollinated. The female cones (seed), which mature in autumn about 18 months after pollination, are globose, large, 18–25 cm (7–10 in) in diameter, and hold about 100–150 seeds. The cones disintegrate at maturity to release the approximately 5 cm (2 in) long nut-like seeds, which are then dispersed by animals, notably the azure jay, Cyanocorax caeruleus.

The inner bark and resin from the trunk of the tree is reddish, which can be a good defining character because it differs from A. araucana, which has brown bark inner and white resin.
###Habitat and ecology

It prefers well drained, slightly acidic soil but will tolerate almost any soil type provided drainage is good. It requires a subtropical/temperate climate with abundant rainfall, tolerating occasional frosts down to about −5 to −20 °C (23 to −4 °F).

The seeds are very important for the native animals. Several mammals and birds eat the pinhão, and it has an important ecological role in Araucaria moist forests (a sub-type of the Brazilian Atlantic Forest).

In a long term study observing the feeding behaviour throughout the year of the squirrel Guerlinguetus brasiliensis ssp. ingrami in a secondary A. angustifolia forest in the Parque Recreativo Primavera in the vicinity of the city of Curitiba, Paraná, of the ten plant species of which the squirrel ate the seeds or nuts, seeds of A. angustifolia were the most important food item in the fall, with a significant percentage of their diet in the winter consisting of the seeds as well.

The squirrels cache seeds, but it is unclear how this affects recruitment.
### Human use

It is a popular garden tree in subtropical areas, planted for its unusual effect of the thick, 'reptilian' branches with a very symmetrical appearance.

The seeds, similar to large pine nuts, are edible, and are extensively harvested in southern Brazil (Paraná, Santa Catarina and Rio Grande do Sul states), an occupation particularly important for the region's small population of natives (the Kaingáng and other Southern Jê). The seeds, called pinhão  are popular as a winter snack. The city of Lages, in Santa Catarina, holds a popular pinhão fair, in which mulled wine and boiled Araucaria seeds are consumed. 3,400 tonnes (7,500,000 lb) of seeds are collected annually in Brazil.

It is also used as a softwood timber in stair treads and joinery. The species is widely used in folk medicine.

A. angustifolia is grown as an ornamental plant in parks of towns and cities of Chile, from Santiago to Valdivia. It grows better in low altitudes than the local Araucaria araucana, hence its use as a substitute in the Central Valley and coastal regions of Chile. In some places like the town of Melipeuco A. angustifolia can be seen growing side by side with A. araucana.

The hybrid Araucaria angustifolia × araucana is thought to have first arisen "in a plantation forestry environment in Argentina sometime in the late 19th or early 20th century". It is thus not a natural hybrid as there are more than 1000 km between the natural stands of the two species.
Role in forest expansion

Studies show the crucial contribution of Araucaria trees in promoting forest expansion. Araucaria angustifolia trees play a pivotal role in shaping the landscape and fostering ecological diversity in southern Brazilian highlands. These conifers act as a facilitator species, also known as nurse trees, significantly increasing species richness and abundance of other trees beneath their crowns. The crowns of these iconic trees foster a unique microenvironment that positively influences the structure and diversity of plant communities 
### Conservation

According to one calculation it has lost an estimated 97% of its habitat to logging, agriculture, and silviculture in the last century. People also eat the seeds, which may reduce recruitment. It was therefore listed by the IUCN as 'vulnerable in 1998 and 'critically endangered' in 2008. 
'''),
    SpeciesDetail('Aspidosperma polyneuron', '''
Aspidosperma polyneuron is a timber tree native to Brazil, Colombia, Peru, Argentina, and Paraguay. It is common in Atlantic Forest vegetation. In addition, it is useful for beekeeping. 
''', "https://en.wikipedia.org/wiki/Aspidosperma_polyneuron", '''
'''),
    SpeciesDetail('Bagassa guianensis', '''
Bagassa guianensis is a tree in the plant family Moraceae which is native to the Guianas and Brazil. It is valued as a timber tree and as a food tree for wildlife. The juvenile leaves are distinctly different in appearance from the mature leaves, and were once thought to belong to different species. 
''', "https://en.wikipedia.org/wiki/Bagassa", '''
Bagassa guianensis is a tree in the plant family Moraceae which is native to the Guianas and Brazil. It is valued as a timber tree and as a food tree for wildlife. The juvenile leaves are distinctly different in appearance from the mature leaves, and were once thought to belong to different species.
###Description

Bagassa guianensis is a large, latex-producing, dioecious, deciduous tree which reaches heights of up 45 metres (148 feet) and a diameter at breast height of 190 centimetres (75 inches). The leaves are deeply three-lobed in juveniles, but become entire as the tree matures. They are usually 6–22 cm (2+1⁄4–8+3⁄4 in) long, sometimes up to 30 cm (12 in) long, and 4–17 cm (1+1⁄2–6+3⁄4 in) wide (sometimes up to 23 cm (9 in) wide).

Male and female flowers are borne on separate inflorescences. Male inflorescences are arranged in a spike, which is 4–12 cm (1+1⁄2–4+3⁄4 in) long. Female inflorescences are arranged into a compact head which is 1 to 1.5 cm (3⁄8 to 5⁄8 in) in diameter. The infructescences are 2.5 to 3.5 cm (1 to 1+1⁄2 in) in diameter.
###Taxonomy

Bagassa is a monotypic genus—it includes only one species, B. guianensis. The genus was established in 1775 by French botanist Jean Baptiste Christophore Fusée Aublet in his description of the species. Aublet's description was based on juvenile leaves together with infructescences. Based on mature leaves and male inflorescences, French botanist Nicaise Auguste Desvaux described Piper tiliifolium in 1825 and Charles Gaudichaud-Beaupré described Laurea tiliifolia in 1844. Raymond Benoist transferred these to Bagassa as B. tiliifolia in 1933. In 1880 Louis Édouard Bureau described B. sagotiana based on mature leaves and female inflorescences. Plants with juvenile and adult foliage were thought to belong to different species until at least 1975; in his 1975 treatment of the Moraceae for the Flora of Suriname, Dutch systematist Cornelis Berg maintained B. guianensis and B. tiliifolia as separate species—the former with lobed juvenile foliage, the latter with the entire leaves of mature trees (although he maintained this distinction with reservations). This confusion would later be clarified through observations of live trees in the field.
###Common names

The species is known locally as "cow wood", katowar, tuwue or yawahedan[what language is this?] in Guyana. In Suriname is it known as gele bagasse, jawahedan, kauhoedoe or kaw-oedoe. In French Guiana it is called bacasse, bagasse, odon or odoun. In Maranhão state in Brazil it is called tatajuba or tareka'y; in Pará it is known as amaparana, taraiko'i or tatajuba; in Roraima it is called tatajuba.[what language is this?]
###Distribution

Bagassa guianensis is found in Guyana, Suriname, French Guiana and the northern Amazon basin (in the states of Amapá, Pará, Maranhão and Roraima) with an apparently disjunct population in the southwestern states of Mato Grosso and Rondônia.
###Ecology

Bagassa guianensis is a "long-lived pioneer" that frequently established in second growth forests and tree-fall gaps.

Although the structure of B. guianensis flowers suggests bat-pollination, Berg suggested that they might be wind-pollinated since the trees were "tall and deciduous". Direct observation suggests that pollination is primarily by thrips, although the thrips themselves may be dispersed by wind. One study in Pará, Brazil, suggests that on average, seeds were produced by pollen that had travelled between 308 and 961 m (1,010 and 3,153 ft) from the male flowers that produced the pollen to the female flowers that were pollinated.

The seeds of B. guianensis are dispersed by a variety of animals including monkeys, birds, deer, rodents and tortoises.
###Uses

Bagassa guianensis is a valuable timber species and is intensively exploited. It is used for construction, furniture, and boat-building.

The infructescences are edible.

'''),
    SpeciesDetail('Balfourodendron riedelianum', '''
Balfourodendron riedelianum, known as marfim in Portuguese, is a species of flowering tree in the rue family, Rutaceae. It is native to Argentina, Brazil, and Paraguay. 
''', "https://en.wikipedia.org/wiki/Balfourodendron_riedelianum", '''
'''),
    SpeciesDetail('Bertholethia excelsa', '''
The Brazil nut (Bertholletia excelsa) is a South American tree in the family Lecythidaceae, and it is also the name of the tree's commercially harvested edible seeds. It is one of the largest and longest-lived trees in the Amazon rainforest. The fruit and its nutshell - containing the edible Brazil nut - are relatively large, possibly weighing as much as 2 kg (4.4 lb) in total weight. As food, Brazil nuts are notable for diverse content of micronutrients, especially a high amount of selenium. The wood of the Brazil nut tree is prized for its quality in carpentry, flooring, and heavy construction. 
''', "https://en.wikipedia.org/wiki/Brazil_nut", '''
The Brazil nut (Bertholletia excelsa) is a South American tree in the family Lecythidaceae, and it is also the name of the tree's commercially harvested edible seeds. It is one of the largest and longest-lived trees in the Amazon rainforest. The fruit and its nutshell – containing the edible Brazil nut – are relatively large, possibly weighing as much as 2 kg (4.4 lb) in total weight. As food, Brazil nuts are notable for diverse content of micronutrients, especially a high amount of selenium. The wood of the Brazil nut tree is prized for its quality in carpentry, flooring, and heavy construction.
###Common names

In Portuguese-speaking countries, like Brazil, they are variously called "castanha-do-brasil" (meaning "chestnuts from Brazil" in Portuguese), "castanha-do-pará" (meaning "chestnuts from Pará" in Portuguese), with other names: castanha-da-amazônia, castanha-do-acre, "noz amazônica" (meaning "Amazonian nut" in Portuguese), noz boliviana, tocari ("probably of Carib origin"), and tururi (from Tupi turu'ri) also used.

In various Spanish-speaking countries of South America, Brazil nuts are called castañas de Brasil, nuez de Brasil, or castañas de Pará (or Para).

In North America, as early as 1896, Brazil nuts were sometimes known by the slang term "nigger toes", a vulgarity that fell out of use after the racial slur became socially unacceptable.
###Description

The Brazil nut is a large tree, reaching 50 metres (160 feet) tall, and with a trunk 1 to 2 m (3 to 7 ft) in diameter, making it among the largest of trees in the Amazon rainforest. It may live for 500 years or more, and can often reach a thousand years of age. The stem is straight and commonly without branches for well over half the tree's height, with a large, emergent crown of long branches above the surrounding canopy of other trees.

The bark is grayish and smooth. The leaves are dry-season deciduous, alternate, simple, entire or crenate, oblong, 20–35 centimetres (8–14 inches) long, and 10–15 cm (4–6 in) broad. The flowers are small, greenish-white, in panicles 5–10 cm (2–4 in) long; each flower has a two-parted, deciduous calyx, six unequal cream-colored petals, and numerous stamens united into a broad, hood-shaped mass.
###Reproduction

Brazil nut trees produce fruit almost exclusively in pristine forests, as disturbed forests lack the large-bodied bees of the genera Bombus, Centris, Epicharis, Eulaema, and Xylocopa, which are the only ones capable of pollinating the tree's flowers, with different bee genera being the primary pollinators in different areas, and different times of year. Brazil nuts have been harvested from plantations, but production is low and is currently not economically viable.

The fruit takes 14 months to mature after pollination of the flowers. The fruit itself is a large capsule 10–15 cm (4–6 in) in diameter, resembling a coconut endocarp in size and weighing up to 2 kg (4 lb 7 oz). It has a hard, woody shell 8–12 mm (3⁄8–1⁄2 in) thick, which contains eight to 24 wedge-shaped seeds 4–5 cm (1+5⁄8–2 in) long (the "Brazil nuts") packed like the segments of an orange, but not limited to one whorl of segments. Up to three whorls can be stacked onto each other, with the polar ends of the segments of the middle whorl nestling into the upper and lower whorls (see illustration above).

The capsule contains a small hole at one end, which enables large rodents like the agouti to gnaw it open. They then eat some of the seeds inside while burying others for later use; some of these are able to germinate into new Brazil nut trees. Most of the seeds are "planted" by the agoutis in caches during wet season, and the young saplings may have to wait years, in a state of dormancy, for a tree to fall and sunlight to reach it, when it starts growing again. Capuchin monkeys have been reported to open Brazil nuts using a stone as an anvil.
###Taxonomy

The Brazil nut family, the Lecythidaceae, is in the order Ericales, as are other well-known plants such as blueberries, cranberries, sapote, gutta-percha, tea, phlox, and persimmons. The tree is the only species in the monotypic genus Bertholletia, named after French chemist Claude Louis Berthollet.
###Distribution and habitat

The Brazil nut is native to the Guianas, Venezuela, Brazil, eastern Colombia, eastern Peru, and eastern Bolivia. It occurs as scattered trees in large forests on the banks of the Amazon River, Rio Negro, Tapajós, and the Orinoco. The fruit is heavy and rigid; when the fruits fall, they pose a serious threat to vehicles and potential for traumatic brain injury of people passing under the tree.
###Production

In 2020, global production of Brazil nuts (in shells) was 69,658 tonnes, most of which derive from wild harvests in tropical forests, especially the Amazon regions of Brazil and Bolivia which produced 92% of the world total (table).
####Environmental effects of harvesting

Since most of the production for international trade is harvested in the wild, the business arrangement has been advanced as a model for generating income from a tropical forest without destroying it. The nuts are most often gathered by migrant workers known as castañeros (in Spanish) or castanheiros (in Portuguese). Logging is a significant threat to the sustainability of the Brazil nut-harvesting industry.

Analysis of tree ages in areas that are harvested shows that moderate and intense gathering takes so many seeds that not enough are left to replace older trees as they die. Sites with light gathering activities had many young trees, while sites with intense gathering practices had nearly none.
####European Union import regulation

In 2003, the European Union imposed strict regulations on the import of Brazilian-harvested Brazil nuts in their shells, as the shells are considered to contain unsafe levels of aflatoxins, a potential cause of liver cancer.
###Toxicity

Brazil nuts are susceptible to contamination by aflatoxins, produced by fungi, once they fall to the ground. Aflatoxins can cause liver damage, including possible cancer, if consumed. Aflatoxin levels have been found in Brazil nuts during inspections that were far higher than the limits set by the EU. However, mechanical sorting and drying was found to eliminate 98% of aflatoxins; a 2003 EU ban on importation was rescinded after new tolerance levels were set.

The nuts often contain radium, a radioactive element, with a kilogram of nuts containing an activity between 40 and 260 becquerels (1 and 7 nanocuries). This level of radium is small, although it can be about 1,000 times higher than in other common foods. According to Oak Ridge Associated Universities, elevated levels of radium in the soil does not directly cause the concentration of radium, but "the very extensive root system of the tree" can concentrate naturally occurring radioactive material, when present in the soil. Radium can be concentrated in nuts only if it is present in the soil.

Brazil nuts also contain barium, a metal with a chemical behavior quite similar to radium. While barium, if ingested, can have toxic effects, such as weakness, vomiting, or diarrhea, the amount present in Brazil nuts are orders of magnitude too small to have noticeable health effects.
###Uses

Brazil nuts after shell removal
####Nutrition

Brazil nuts are 3% water, 14% protein, 12% carbohydrates, and 66% fats (table). The fat components are 16% saturated, 24% monounsaturated, and 24% polyunsaturated (see table for USDA source).

In a 100 grams (3.5 ounces) reference amount, Brazil nuts supply 659 calories, and are a rich source (20% or more of the Daily Value, DV) of dietary fiber (30% DV), thiamin (54% DV), vitamin E (38% DV), magnesium (106% DV), phosphorus (104% DV), manganese (57% DV), and zinc (43% DV). Calcium, iron, and potassium are present in moderate amounts (10-19% DV, table).
####Selenium

Brazil nuts are a particularly rich source of selenium, with just 28 g (1 oz) supplying 544 micrograms of selenium or 10 times the DV of 55 micrograms (see table for USDA source). However, the amount of selenium within batches of nuts may vary considerably.

The high selenium content is used as a biomarker in studies of selenium intake and deficiency. Consumption of just one Brazil nut per day over 8 weeks was sufficient to restore selenium blood levels and increase HDL cholesterol in obese women.
####Oil

Brazil nut oil contains 48% unsaturated fatty acids composed mainly of oleic and linoleic acids, the phytosterol, beta-sitosterol, and fat-soluble vitamin E.

The lumber from Brazil nut trees (not to be confused with Brazilwood) is of excellent quality, having diverse uses from flooring to heavy construction. Logging the trees is prohibited by law in all three producing countries (Brazil, Bolivia, and Peru). Illegal extraction of timber and land clearances present continuing threats. In Brazil, cutting down a Brazil nut tree requires previous authorization from the Brazilian Institute of Environment and Renewable Natural Resources.
####Other uses

Brazil nut oil is used as a lubricant in clocks, in the manufacturing of paint and cosmetics, such as soap and perfume. Because of its hardness, the Brazil nutshell is often pulverized and used as an abrasive to polish materials, such as metals and ceramics, in the same way as a jeweler's rouge. The charcoal from the nut shells may be used to purify water. 
'''),
    SpeciesDetail('Bowdichia sp', '''
Bowdichia is a genus of flowering plants in the legume family, Fabaceae. It belongs to the subfamily Faboideae. The genus includes two species native to tropical South America and Costa Rica.
Bowdichia nitida Spruce ex Benth. – northern Brazil and northern Bolivia
Bowdichia virgilioides Kunth – Costa Rica to Bolivia, Paraguay, and southern Brazil
''', "https://en.wikipedia.org/wiki/Bowdichia", '''
'''),
    SpeciesDetail('Brosimum paraense', '''
Brosimum is a genus of plants in the family Moraceae, native to tropical regions of the Americas.
The breadnut (B. alicastrum) was used by the Maya civilization for its edible nut. The dense vividly colored scarlet wood of B. paraense is used for decorative woodworking. B. guianense, or snakewood, has a mottled snake-skin pattern, and is among the densest woods, with a very high stiffness; it was the wood of choice for making of bows for musical instruments of the violin family until the late 18th century, when it was replaced by the more easily worked brazilwood (Paubrasilia echinata). Plants of this genus are otherwise used for timber, building materials, and in a cultural context. 
''', "https://en.wikipedia.org/wiki/Brosimum", '''
'''),
    SpeciesDetail('Carapa guianensis', '''
Carapa guianensis is a species of tree in the family Meliaceae, also known by the common names andiroba or crabwood.
Andiroba is native to the Amazon and is widely used by the indigenous populations of the northern region of Brazil. It grows in the Amazon region, Central America and the Caribbean. It is a tall tree with dense foliage and usually grows in the tropical rainforest along the edge of rivers. 
''', "https://en.wikipedia.org/wiki/Carapa_guianensis", '''
Carapa guianensis is a species of tree in the family Meliaceae, also known by the common names andiroba or crabwood.
###Description

Andiroba is native to the Amazon and is widely used by the indigenous populations of the northern region of Brazil. It grows in the Amazon region, Central America and the Caribbean. It is a tall tree with dense foliage and usually grows in the tropical rainforest along the edge of rivers.
###Uses

The timber is used in furniture and flooring. While the wood is not classified as genuine mahogany, it is related to the mahogany family and is similar in appearance.

The oil contained in the andiroba almond, known as crab oil or carap oil, is light yellow and extremely bitter. When subjected to a temperature below 25 °C, it solidifies, with a consistency like that of petroleum jelly. It contains olein, palmitin and glycerin.

The oil and fats of the almond are extracted and used for the production of insect repellent and compounds for traditional medicine. It is used in Brazil to protect furniture from termites and other wood-chewing insects. 
'''),
    SpeciesDetail('Cariniana estrellensis', '''
Cariniana estrellensis is a species of rainforest tree in the family Lecythidaceae. It is native to South America. These trees can grow to extraordinary size. Perhaps the largest rainforest tree ever measured by college trained forester was a C. estrellensis measured by Edmundo Navarro de Andrade which was twenty-three feet thick (twenty-two meters girth) with no buttresses or basal swelling. 
''', "https://en.wikipedia.org/wiki/Cariniana_estrellensis", '''
'''),
    SpeciesDetail('Cedrela fissilis', '''
Cedrela fissilis is a species of tree in the family Meliaceae. It is native to Central and South America, where it is distributed from Costa Rica to Argentina. Its common names include Argentine cedar, cedro batata, cedro blanco, "Acaju-catinga" (its Global Trees entry) and cedro colorado.
Once a common lowland forest tree, this species has been overexploited for timber and is now considered to be endangered. A few populations are stable, but many have been reduced, fragmented, and extirpated. The wood is often sold in batches with Cuban cedar (Cedrela odorata). 
''', "https://en.wikipedia.org/wiki/Cedrela_fissilis", '''
'''),
    SpeciesDetail('Cedrelinga catenaeformis', '''
Cedrelinga is a genus of trees in the family Fabaceae. The only accepted species is Cedrelinga cateniformis, called tornillo or cedrorana, which is native to South America. It is occasionally harvested for its straight-grained timber.
''', "https://en.wikipedia.org/wiki/Cedrelinga", '''
'''),
    SpeciesDetail('Cordia goeldiana', '''

###Botanical Description

It is often about 10 to 20 m in height, with a trunk diameter of about 40 to 60 cm. Heights of over 30 m and trunk diameters of up to 90 cm are also possible.
###Natural Habitat

Cordia goeldiana is usually found in the lower Amazon in terra firme forests. It is reported in primary forests, although it also develops well under exposed conditions.
###Natural Distribution
                  
Freijo is reported to occur in Para and in the Tocantins and Xingu River basins of Brazil.
''', "http://www.tropicaltimber.info/specie/freijo-cordia-goeldiana/", '''
'''),
    SpeciesDetail('Cordia sp', '''
Cordia is a genus of flowering plants in the borage family, Boraginaceae. It contains about 300 species of shrubs and trees, that are found worldwide, mostly in warmer regions. Many of the species are commonly called manjack, while bocote may refer to several Central American species in Spanish.
The generic name honours German botanist and pharmacist Valerius Cordus (1515–1544). Like most other Boraginaceae, the majority have trichomes (hairs) on the leaves. 
''', "https://en.wikipedia.org/wiki/Cordia", '''
Cordia is a genus of flowering plants in the borage family, Boraginaceae. It contains about 300 species of shrubs and trees, that are found worldwide, mostly in warmer regions. Many of the species are commonly called manjack, while bocote may refer to several Central American species in Spanish.

The generic name honours German botanist and pharmacist Valerius Cordus (1515–1544). Like most other Boraginaceae, the majority have trichomes (hairs) on the leaves.
###Taxonomy

The taxonomy of Cordia is complex and controversial. Gottschling et al. (2005) say this is partly due to "extraordinarily high intraspecific variability" in some groups of species, making identification difficult, and partly due to new taxa having been "airily described on the basis of poorly preserved herbarium specimens".
###Ecology

Cordia species are used as food plants by the caterpillars of some Lepidoptera species, such as Endoclita malabaricus, Bucculatrix caribbea, and Bucculatrix cordiaella. The wild olive tortoise beetle (Physonota alutacea) feeds on C. boissieri, C. dentata, C. inermis, and C. macrostachya.
###Uses
####Ornamental

Many members of this genus have fragrant, showy flowers and are popular in gardens, although they are not especially hardy.
####As food

A number of the tropical species have edible fruits, known by a wide variety of names including clammy cherries, glue berries, sebesten, or snotty gobbles. In India, the fruits of local species are used as a vegetable, raw, cooked, or pickled, and are known by many names, including lasora in Hindi. One such species is fragrant manjack (C. dichotoma), which is called gunda or tenti dela in Hindi and lasura in Nepali. The fruit of the fragrant manjack is called phoà-pò·-chí (破布子), 樹子仔, or 樹子(Pe̍h-ōe-jī: chhiū-chí) in Taiwan where they are eaten pickled.
####Glue

The white, gooey inner pulp of the fruits is traditionally used to make glue.
####Wood

The wood of several Cordia species is commercially harvested. Ecuador laurel (C. alliodora), ziricote (C. dodecandra), Spanish elm (C. gerascanthus), and C. goeldiana are used to make furniture and doors in Central and South America.

Ziricote and bocote are sometimes used as tonewoods for making the backs and sides of high-end acoustic guitars such as the Richard Thompson signature model from Lowden. Similarly, drums are made from C. abyssinica, C. millenii, and C. platythyrsa due to the resonance of the wood.
####Smoking

Cordia leaves can be dried and used to smoke marijuana with. 
'''),
    SpeciesDetail('Couratari sp', '''
Couratari is a genus of trees in the family Lecythidaceae, first described as a genus in 1775. They are native to tropical South America and Central America.
They are large trees, often rising above the rainforest canopy. The leaves are evergreen, alternate, simple, elliptical, up to 15 cm long, with a serrate to serrulate margin. Vernation lines parallel to the midvein are often visible - a very unusual characteristic. The fruit is 6–15 cm long, and roughly conical. A central plug drops out at maturity, releasing the winged seeds to be dispersed by wind. The fruit of Cariniana may be distinguished from those of Couratari, as the former have longitudinal ridges, whereas the latter bears a single calyx-derived ring near the fruit apex. 
''', "https://en.wikipedia.org/wiki/Couratari", '''
'''),
    SpeciesDetail('Dipteryx sp', '''
Dipteryx is a genus containing a number of species of large trees and possibly shrubs. It belongs to the "papilionoid" subfamily – Faboideae – of the family Fabaceae. This genus is native to South and Central America and the Caribbean. Formerly, the related genus Taralea was included in Dipteryx. 
''', "https://en.wikipedia.org/wiki/Dipteryx", '''
Dipteryx is a genus containing a number of species of large trees and possibly shrubs. It belongs to the "papilionoid" subfamily – Faboideae – of the family Fabaceae. This genus is native to South and Central America and the Caribbean. Formerly, the related genus Taralea was included in Dipteryx.
###Description

The largest members of Dipteryx are canopy-emergent trees of tropical rainforests. The tonka bean (D. odorata) is grown for its fragrant seeds. Baru (D. alata) is the only species which found in drier, seasonal areas, growing in the cerrado of Brazil; its fruit and seeds are used as food and fodder. Several species are used for timber, of which almendro (D. oleifera) wood is considered desirable, especially locally.

Dipteryx can be distinguished from other members of the Dipterygeae by its compound leaves with asymmetric leaflets caused due to an eccentric primary vein, a drupaceous fruit, seeds with a leathery skin, a hilum in a lateral or subapical position and a rugose embryo with a conspicuous plumule.
###Taxonomy

The number of recognised species of Dipteryx has changed over the years.

The genus was previously known as Coumarouna. In 1934 Walter Adolpho Ducke split this genus into two, on the basis of the alternate leaflets, among other characters, of Dipteryx. He used two older, conserved names published previously: Taralea and Dipteryx. Although Taralea was accepted, some taxonomists did not recognise Dipteryx as the correct name for the genus until at least the mid-1940s.

In the most recent monograph on the genus, A Checklist of the Dipterygeae species by the Brazilian researcher Haroldo Cavalcante de Lima in 1989, he synonymised a number of species, accepting nine species in the genus. His taxonomy was accepted by ILDIS (2005) but not noticed or followed by US databases, i.e. in GRIN (2005), the entry on Dipteryx in the Contribución al conocimiento de las leguminosas Colombianas by C. Barbosa (1994), the IUCN (1998) based on World List of Threatened Trees by Oldfield et al. (1998), or the Catalogue of the Flowering Plants and Gymnosperms of Peru (1993) which was built using the Tropicos database by the Missouri Botanical Garden. In 1999 the entry on Dipteryx in the Flora of the Venezuelan Guyana by de Lima was published.

The northernmost taxon Dipteryx panamensis, notable as being the only species listed on CITES since 2003 and therefore subject to export controls, was synonymised with the neglected but older name D. oleifera by de Lima in 1989, but this move was only followed by ILDIS and one or two of articles on the species over the years, all other floras, databases and publications using the name D. panamensis. In 2011, however, the Report of the Nomenclature Committee for Vascular Plants: 62 recommended D. oleifera by treated as validly published, and de Lima's synonymy for this taxon has been accepted by many.

By 2010, in the Catálogo de Plantas e Fungos do Brasil, de Lima had changed his mind and re-recognised two of Brazilian taxa he had earlier considered synonyms in 1989, although not all. 
'''),
    SpeciesDetail('Erisma uncinatum', '''
Erisma uncinatum is a species of tree in the family Vochysiaceae. They have a self-supporting growth form. They are native to Amapá, Mato Grosso, Maranhao, Amazônia, RondôNia, Pará, Acre (Brazil), and The Neotropics. They have simple, broad leaves. Individuals can grow to 27 m.
''', "https://eol.org/pages/5497350", '''
'''),
    SpeciesDetail('Eucalyptus sp', '''
Eucalyptus (/ˌjuːkəˈlɪptəs/) is a genus of more than 700 species of flowering plants in the family Myrtaceae. Most species of Eucalyptus are trees, often mallees, and a few are shrubs. Along with several other genera in the tribe Eucalypteae, including Corymbia and Angophora, they are commonly known as eucalypts or "gum trees". Plants in the genus Eucalyptus have bark that is either smooth, fibrous, hard or stringy, the leaves have oil glands, and the sepals and petals are fused to form a "cap" or operculum over the stamens. The fruit is a woody capsule commonly referred to as a "gumnut". 
''', "https://en.wikipedia.org/wiki/Eucalyptus", '''
Eucalyptus (/ˌjuːkəˈlɪptəs/) is a genus of more than 700 species of flowering plants in the family Myrtaceae. Most species of Eucalyptus are trees, often mallees, and a few are shrubs. Along with several other genera in the tribe Eucalypteae, including Corymbia and Angophora, they are commonly known as eucalypts or "gum trees". Plants in the genus Eucalyptus have bark that is either smooth, fibrous, hard or stringy, the leaves have oil glands, and the sepals and petals are fused to form a "cap" or operculum over the stamens. The fruit is a woody capsule commonly referred to as a "gumnut".

Most species of Eucalyptus are native to Australia, and every state and territory has representative species. About three-quarters of Australian forests are eucalypt forests. Many eucalypt species have adapted to wildfire, and are able to resprout after fire or have seeds which survive fire.

A few species are native to islands north of Australia and a smaller number are only found outside the continent. Eucalypts have been grown in plantations in many other countries because they are fast growing and have valuable timber, or can be used for pulpwood, for honey production or essential oils. In some countries, however, they have been removed because of the danger of forest fires due to their high flammability.
###Description
####Size and habit

Eucalypts vary in size and habit from shrubs to tall trees. Trees usually have a single main stem or trunk but many eucalypts are mallees that are multistemmed from ground level and rarely taller than 10 metres (33 feet). There is no clear distinction between a mallee and a shrub but in eucalypts, a shrub is a mature plant less than 1 m (3 ft 3 in) tall and growing in an extreme environment. Eucalyptus vernicosa in the Tasmanian highlands, E. yalatensis on the Nullarbor and E. surgens growing on coastal cliffs in Western Australia are examples of eucalypt shrubs.

The terms "mallet" and "marlock" are only applied to Western Australian eucalypts. A mallet is a tree with a single thin trunk with a steeply branching habit but lacks both a lignotuber and epicormic buds. Eucalyptus astringens is an example of a mallet. A marlock is a shrub or small tree with a single, short trunk, that lacks a lignotuber and has spreading, densely leafy branches that often reach almost to the ground. Eucalyptus platypus is an example of a marlock.

Eucalyptus trees, including mallets and marlocks, are single-stemmed and include Eucalyptus regnans, the tallest known flowering plant on Earth. The tallest reliably measured tree in Europe, Karri Knight, can be found in Coimbra, Portugal in Vale de Canas. It is an Eucalyptus diversicolor of 72.9 meters height and of 5.71 meters girth.

Tree sizes follow the convention of:

    Small: to 10 m (33 ft) in height
    Medium-sized: 10–30 m (33–98 ft)
    Tall: 30–60 m (98–197 ft)
    Very tall: over 60 m (200 ft)

####Bark

All eucalypts add a layer of bark every year and the outermost layer dies. In about half of the species, the dead bark is shed exposing a new layer of fresh, living bark. The dead bark may be shed in large slabs, in ribbons or in small flakes. These species are known as "smooth barks" and include E. sheathiana, E. diversicolor, E. cosmophylla and E. cladocalyx. The remaining species retain the dead bark which dries out and accumulates. In some of these species, the fibres in the bark are loosely intertwined (in stringybarks such as E. macrorhyncha or peppermints such as E. radiata) or more tightly adherent (as in the "boxes" such as E. leptophleba). In some species (the "ironbarks" such as E. crebra and E. jensenii) the rough bark is infused with gum resin.

Many species are ‘half-barks’ or ‘blackbutts’ in which the dead bark is retained in the lower half of the trunks or stems—for example, E. brachycalyx, E. ochrophloia, and E. occidentalis—or only in a thick, black accumulation at the base, as in E. clelandii. In some species in this category, for example E. youngiana and E. viminalis, the rough basal bark is very ribbony at the top, where it gives way to the smooth upper stems. The smooth upper bark of the half-barks and that of the completely smooth-barked trees and mallees can produce remarkable colour and interest, for example E. deglupta.

E. globulus bark cells are able to photosynthesize in the absence of foliage, conferring an "increased capacity to re-fix internal CO2 following partial defoliation". This allows the tree to grow in less-than-ideal climates, in addition to providing a better chance of recovery from damage sustained to its leaves in an event such as a fire.

Different commonly recognised types of bark include:

    Stringybark—consists of long fibres and can be pulled off in long pieces. It is usually thick with a spongy texture.
    Ironbark—is hard, rough, and deeply furrowed. It is impregnated with dried kino (a sap exuded by the tree) which gives a dark red or even black colour.
    Tessellated—bark is broken up into many distinct flakes. They are corkish and can flake off.
    Box—has short fibres. Some also show tessellation.
    Ribbon—has the bark coming off in long, thin pieces, but is still loosely attached in some places. They can be long ribbons, firmer strips, or twisted curls.

####Leaves

Nearly all Eucalyptus are evergreen, but some tropical species lose their leaves at the end of the dry season. As in other members of the myrtle family, Eucalyptus leaves are covered with oil glands. The copious oils produced are an important feature of the genus. Although mature Eucalyptus trees may be towering and fully leafed, their shade is characteristically patchy because the leaves usually hang downwards.

The leaves on a mature Eucalyptus plant are commonly lanceolate, petiolate, apparently alternate and waxy or glossy green. In contrast, the leaves of seedlings are often opposite, sessile and glaucous, but many exceptions to this pattern exist. Many species such as E. melanophloia and E. setosa retain the juvenile leaf form even when the plant is reproductively mature. Some species, such as E. macrocarpa, E. rhodantha, and E. crucis, are sought-after ornamentals due to this lifelong juvenile leaf form. A few species, such as E. petraea, E. dundasii, and E. lansdowneana, have shiny green leaves throughout their life cycle. Eucalyptus caesia exhibits the opposite pattern of leaf development to most Eucalyptus, with shiny green leaves in the seedling stage and dull, glaucous leaves in mature crowns. The contrast between juvenile and adult leaf phases is valuable in field identification.

Four leaf phases are recognised in the development of a Eucalyptus plant: the ‘seedling’, ‘juvenile’, ‘intermediate’, and ‘adult’ phases. However, no definite transitional point occurs between the phases. The intermediate phase, when the largest leaves are often formed, links the juvenile and adult phases.

In all except a few species, the leaves form in pairs on opposite sides of a square stem, consecutive pairs being at right angles to each other (decussate). In some narrow-leaved species, for example E. oleosa, the seedling leaves after the second leaf pair are often clustered in a detectable spiral arrangement about a five-sided stem. After the spiral phase, which may last from several to many nodes, the arrangement reverts to decussate by the absorption of some of the leaf-bearing faces of the stem. In those species with opposite adult foliage the leaf pairs, which have been formed opposite at the stem apex, become separated at their bases by unequal elongation of the stem to produce the apparently alternate adult leaves.
####Flowers and fruits

The most readily recognisable characteristics of Eucalyptus species are the distinctive flowers and fruit (capsules or "gumnuts"). Flowers have numerous fluffy stamens which may be white, cream, yellow, pink, or red; in bud, the stamens are enclosed in a cap known as an operculum which is composed of the fused sepals or petals, or both. Thus, flowers have no petals, but instead decorate themselves with the many showy stamens. As the stamens expand, the operculum is forced off, splitting away from the cup-like base of the flower; this is one of the features that unites the genus. The woody fruits or capsules are roughly cone-shaped and have valves at the end which open to release the seeds, which are waxy, rod-shaped, about 1 mm in length, and yellow-brown in colour. Most species do not flower until adult foliage starts to appear; E. cinerea and E. perriniana are notable exceptions.
###Taxonomy

The genus Eucalyptus was first formally described in 1789 by Charles Louis L'Héritier de Brutelle who published the description in his book Sertum Anglicum, seu, Plantae rariores quae in hortis juxta Londinum along with a description of the type species, Eucalyptus obliqua. The name Eucalyptus is derived from the Ancient Greek words eu meaning 'good': 373  and kalypto meaning '(I) cover, conceal, hide': 234  referring to the operculum covering the flower buds.

The type specimen was collected in 1777 by David Nelson, the gardener-botanist on Cook's third voyage. He collected the specimen on Bruny Island and sent it to de Brutelle who was working in London at that time.
###History

Although eucalypts must have been seen by the very early European explorers and collectors, no botanical collections of them are known to have been made until 1770 when Joseph Banks and Daniel Solander arrived at Botany Bay with Captain James Cook. There they collected specimens of E. gummifera and later, near the Endeavour River in northern Queensland, E. platyphylla; neither of these species was named as such at the time.

In 1777, on Cook's third expedition, David Nelson collected a eucalypt on Bruny Island in southern Tasmania. This specimen was taken to the British Museum in London, and was named Eucalyptus obliqua by the French botanist L'Héritier, who was working in London at the time. He coined the generic name from the Greek roots eu and calyptos, meaning "well" and "covered" in reference to the operculum of the flower bud which protects the developing flower parts as the flower develops and is shed by the pressure of the emerging stamens at flowering. It was most likely an accident that L'Héritier chose a feature common to all eucalypts.

The name obliqua was derived from the Latin obliquus, meaning "oblique", which is the botanical term describing a leaf base where the two sides of the leaf blade are of unequal length and do not meet the petiole at the same place.

E. obliqua was published in 1788–89, which coincided with the European colonisation of Australia. Between then and the turn of the 19th century, several more species of Eucalyptus were named and published. Most of these were by the English botanist James Edward Smith and most were, as might be expected, trees of the Sydney region. These include the economically valuable E. pilularis, E. saligna and E. tereticornis.

The first endemic Western Australian Eucalyptus to be collected and subsequently named was the Yate (E. cornuta) by the French botanist Jacques Labillardière, who collected in what is now the Esperance area in 1792.

Several Australian botanists were active during the 19th century, particularly Ferdinand von Mueller, whose work on eucalypts contributed greatly to the first comprehensive account of the genus in George Bentham's Flora Australiensis in 1867. The account is the most important early systematic treatment of the genus. Bentham divided it into five series whose distinctions were based on characteristics of the stamens, particularly the anthers (Mueller, 1879–84), work elaborated by Joseph Henry Maiden (1903–33) and still further by William Faris Blakely (1934). The anther system became too complex to be workable and more recent systematic work has concentrated on the characteristics of buds, fruits, leaves and bark.
####Species and hybrids

Over 700 species of Eucalyptus are known. Some have diverged from the mainstream of the genus to the extent that they are quite isolated genetically and are able to be recognised by only a few relatively invariant characteristics. Most, however, may be regarded as belonging to large or small groups of related species, which are often in geographical contact with each other and between which gene exchange still occurs. In these situations, many species appear to grade into one another, and intermediate forms are common. In other words, some species are relatively fixed genetically, as expressed in their morphology, while others have not diverged completely from their nearest relatives.

Hybrid individuals have not always been recognised as such on first collection and some have been named as new species, such as E. chrysantha (E. preissiana × E. sepulcralis) and E. "rivalis" (E. marginata × E. megacarpa). Hybrid combinations are not particularly common in the field, but some other published species frequently seen in Australia have been suggested to be hybrid combinations. For example, Eucalyptus × erythrandra is believed to be E. angulosa × E. teraptera and due to its wide distribution is often referred to in texts.

Renantherin, a phenolic compound present in the leaves of some Eucalyptus species, allows chemotaxonomic discrimination in the sections renantheroideae and renantherae and the ratio of the amount of leucoanthocyanins varies considerably in certain species.
####Related genera

Eucalyptus is one of three similar genera that are commonly referred to as "eucalypts", the others being Corymbia and Angophora. Many species, though by no means all, are known as gum trees because they exude copious kino from any break in the bark (e.g., scribbly gum). The generic name is derived from the Greek words ευ (eu) "well" and καλύπτω (kalýpto) "to cover", referring to the operculum on the calyx that initially conceals the flower.
###Distribution

There are more than 700 species of Eucalyptus and most are native to Australia; a very small number are found in adjacent areas of New Guinea and Indonesia. One species, Eucalyptus deglupta, ranges as far north as the Philippines. Of the 15 species found outside Australia, just nine are exclusively non-Australian. Species of Eucalyptus are cultivated widely in the tropical and temperate world, including the Americas, Europe, Africa, the Mediterranean Basin, the Middle East, China, and the Indian subcontinent. However, the range over which many eucalypts can be planted in the temperate zone is constrained by their limited cold tolerance.

Australia is covered by 92,000,000 hectares (230,000,000 acres) of eucalypt forest, comprising three quarters of the area covered by native forest. The Blue Mountains of southeastern Australia have been a centre of eucalypt diversification; their name is in reference to the blue haze prevalent in the area, believed derived from the volatile terpenoids emitted by these trees.
###Fossil record

The oldest definitive Eucalyptus fossils are from Patagonia in South America, where eucalypts are no longer native, though they have been introduced from Australia. The fossils are from the early Eocene (51.9 Mya), and were found in the Laguna del Hunco Formation in Chubut Province in Argentina. This shows that the genus had a Gondwanan distribution. Fossil leaves also occur in the Miocene of New Zealand, where the genus is not native today, but again have been introduced from Australia.

Despite the prominence of Eucalyptus in modern Australia, estimated to contribute some 75% of the modern vegetation, the fossil record is very scarce throughout much of the Cenozoic, and suggests that this rise to dominance is a geologically more recent phenomenon. The oldest reliably dated macrofossil of Eucalyptus is a 21-million-year-old tree-stump encased in basalt in the upper Lachlan Valley in New South Wales. Other fossils have been found, but many are either unreliably dated or else unreliably identified.

It is useful to consider where Eucalyptus fossils have not been found. Extensive research has gone into the fossil floras of the Paleocene to Oligocene of South-Eastern Australia, and has failed to uncover a single Eucalyptus specimen. Although the evidence is sparse, the best hypothesis is that in the mid-Tertiary, the continental margins of Australia only supported more mesic noneucalypt vegetation, and that eucalypts probably contributed to the drier vegetation of the arid continental interior. With the progressive drying out of the continent since the Miocene, eucalypts were displaced to the continental margins, and much of the mesic and rainforest vegetation that was once there was eliminated.

The current superdominance of Eucalyptus in Australia may be an artefact of human influence on its ecology. In more recent sediments, numerous findings of a dramatic increase in the abundance of Eucalyptus pollen are associated with increased charcoal levels. Though this occurs at different rates throughout Australia, it is compelling evidence for a relationship between the artificial increase of fire frequency with the arrival of Aboriginals and increased prevalence of this exceptionally fire-tolerant genus.
###Tall timber

Several eucalypt species are among the tallest trees in the world. Eucalyptus regnans, the Australian 'mountain ash', is the tallest of all flowering plants (angiosperms); today, the tallest measured specimen named Centurion is 100.5 m (330 ft) tall. Coast Douglas-fir is about the same height; only coast redwood is taller, and they are conifers (gymnosperms). Six other eucalypt species exceed 80 metres in height: Eucalyptus obliqua, Eucalyptus delegatensis, Eucalyptus diversicolor, Eucalyptus nitens, Eucalyptus globulus and Eucalyptus viminalis.
###Frost intolerance

Most eucalypts are not tolerant of severe cold. Eucalypts do well in a range of climates but are usually damaged by anything beyond a light frost of −5 °C (23 °F); the hardiest are the snow gums, such as Eucalyptus pauciflora, which is capable of withstanding cold and frost down to about −20 °C (−4 °F). Two subspecies, E. pauciflora subsp. niphophila and E. pauciflora subsp. debeuzevillei in particular are even hardier and can tolerate even quite severe winters. Several other species, especially from the high plateau and mountains of central Tasmania such as Eucalyptus coccifera, Eucalyptus subcrenulata and Eucalyptus gunnii, have also produced extreme cold-hardy forms and it is seed procured from these genetically hardy strains that are planted for ornament in colder parts of the world.
###Animal relationships
Further information: List of Lepidoptera that feed on Eucalyptus

An essential oil extracted from Eucalyptus leaves contains compounds that are powerful natural disinfectants and can be toxic in large quantities. Several marsupial herbivores, notably koalas and some possums, are relatively tolerant of it. The close correlation of these oils with other more potent toxins called formylated phloroglucinol compounds (euglobals, macrocarpals and sideroxylonals) allows koalas and other marsupial species to make food choices based on the smell of the leaves. For koalas, these compounds are the most important factor in leaf choice.

Eucalyptus flowers produce a great abundance of nectar, providing food for many pollinators including insects, birds, bats and possums. Many lizard species in Australia feed on Eucalyptus sap as well, famously in the case of the Dubious dtella. Although Eucalyptus trees are seemingly well-defended from herbivores by the oils and phenolic compounds, they have insect pests. These include the eucalyptus longhorn borer Phoracantha semipunctata and the aphid-like psyllids that create "bell lerps", both of which have become established as pests throughout the world wherever eucalypts are cultivated.

The eusocial beetle Austroplatypus incompertus makes and defends its galleries exclusively inside eucalypts, including some species of Eucalyptus and Corymbia.

The trunks and branches of the Eucalyptus tree allow the largest known moth, Zelotypia stacyi (the bentwing ghost moth, having a wingspan up to 250 mm) to feed and protect their larvae and pupae, respectively.

###Diseases on plants

Fungal species Mycosphaerella and Teratosphaeria have been associated with leaf disease on various Eucalyptus species.  Several fungal species from Teratosphaeriaceae family are causal agents in leaf diseases and stem cankers of Eucalyptus in Uruguay, and Australia.
###Adaptation to fire

Eucalypts originated between 35 and 50 million years ago, not long after Australia-New Guinea separated from Gondwana, their rise coinciding with an increase in fossil charcoal deposits (suggesting that fire was a factor even then), but they remained a minor component of the Tertiary rainforest until about 20 million years ago, when the gradual drying of the continent and depletion of soil nutrients led to the development of a more open forest type, predominantly Casuarina and Acacia species.

The two valuable timber trees, alpine ash E. delegatensis and Australian mountain ash E. regnans, are killed by fire and only regenerate from seed. The same 2003 bushfire that had little impact on forests around Canberra resulted in thousands of hectares of dead ash forests. However, a small amount of ash survived and put out new ash trees as well.
####Fire hazard

Eucalyptus oil is highly flammable and vapours from the tree are known to explode in a fire. Bushfires can travel easily through the oil-rich air of the tree crowns. Eucalypts obtain long-term fire survivability from their ability to regenerate from epicormic buds situated deep within their thick bark, or from lignotubers, or by producing serotinous fruits.

In seasonally dry climates oaks are often fire-resistant, particularly in open grasslands, as a grass fire is insufficient to ignite the scattered trees. In contrast, a Eucalyptus forest tends to promote fire because of the volatile and highly combustible oils produced by the leaves, as well as the production of large amounts of litter high in phenolics, preventing its breakdown by fungi and thus accumulating as large amounts of dry, combustible fuel. Consequently, dense eucalypt plantings may be subject to catastrophic firestorms. In fact, almost thirty years before the Oakland firestorm of 1991, a study of Eucalyptus in the area warned that the litter beneath the trees builds up very rapidly and should be regularly monitored and removed. It has been estimated that 70% of the energy released through the combustion of vegetation in the Oakland fire was due to Eucalyptus. In a National Park Service study, it was found that the fuel load (in tons per acre) of non-native Eucalyptus woods is almost three times as great as native oak woodland.

During World War II, one California town cut down their Eucalyptus trees to "about a third of their height in the vicinity of anti-aircraft guns" because of the known fire-fueling qualities of the trees, with the mayor telling a newspaper reporter, "If a shell so much as hits a leaf, it’s supposed to explode."
###Falling branches

Some species of Eucalyptus drop branches unexpectedly. In Australia, Parks Victoria warns campers not to camp under river red gums. Some councils in Australia such as Gosnells, Western Australia, have removed eucalypts after reports of damage from dropped branches, even in the face of lengthy, well publicised protests to protect particular trees. A former Australian National Botanic Gardens director and consulting arborist, Robert Boden, has been quoted referring to "summer branch drop". Dropping of branches is recognised in Australia literature through the fictional death of Judy in Seven Little Australians. Although all large trees can drop branches, the density of Eucalyptus wood is high due to its high resin content, increasing the hazard.
###Cultivation and uses

Eucalypts were introduced from Australia to the rest of the world following the Cook expedition in 1770. Collected by Sir Joseph Banks, botanist on the expedition, they were subsequently introduced to many parts of the world, notably California, southern Europe, Africa, the Middle East, South Asia and South America. About 250 species are under cultivation in California. In Portugal and also Spain, eucalypts have been grown in plantations for the production of pulpwood. Eucalyptus are the basis for several industries, such as sawmilling, pulp, charcoal and others. Several species have become invasive and are causing major problems for local ecosystems, mainly due to the absence of wildlife corridors and rotations management.

Eucalypts have many uses which have made them economically important trees, and they have become a cash crop in poor areas such as Timbuktu, Mali: 22  and the Peruvian Andes, despite concerns that the trees are invasive in some environments like those of South Africa. Best-known are perhaps the varieties karri and yellow box. Due to their fast growth, the foremost benefit of these trees is their wood. They can be chopped off at the root and grow back again. They provide many desirable characteristics for use as ornament, timber, firewood and pulpwood. Eucalyptus wood is also used in a number of industries, from fence posts (where the oil-rich wood's high resistance to decay is valued) and charcoal to cellulose extraction for biofuels. Fast growth also makes eucalypts suitable as windbreaks and to reduce erosion.

Some Eucalyptus species have attracted attention from horticulturists, global development researchers, and environmentalists because of desirable traits such as being fast-growing sources of wood, producing oil that can be used for cleaning and as a natural insecticide, or an ability to be used to drain swamps and thereby reduce the risk of malaria. Eucalyptus oil finds many uses like in fuels, fragrances, insect repellence and antimicrobial activity. Eucalyptus trees show allelopathic effects; they release compounds which inhibit other plant species from growing nearby. Outside their natural ranges, eucalypts are both lauded for their beneficial economic impact on poor populations: 22  and criticised for being "water-guzzling" aliens, leading to controversy over their total impact.

Eucalypts draw a tremendous amount of water from the soil through the process of transpiration. They have been planted (or re-planted) in some places to lower the water table and reduce soil salination. Eucalypts have also been used as a way of reducing malaria by draining the soil in Algeria, Lebanon, Sicily, elsewhere in Europe, in the Caucasus (Western Georgia), and California. Drainage removes swamps which provide a habitat for mosquito larvae, but can also destroy ecologically productive areas. This drainage is not limited to the soil surface, because the Eucalyptus roots are up to 2.5 m (8 ft 2 in) in length and can, depending on the location, even reach the phreatic zone.
###Pulpwood

Eucalyptus is the most common short fibre source for pulpwood to make pulp. The types most often used in papermaking are Eucalyptus globulus (in temperate areas) and the Eucalyptus urophylla x Eucalyptus grandis hybrid (in the tropics). The fibre length of Eucalyptus is relatively short and uniform with low coarseness compared with other hardwoods commonly used as pulpwood. The fibres are slender, yet relatively thick walled. This gives uniform paper formation and high opacity that are important for all types of fine papers. The low coarseness is important for high quality coated papers. Eucalyptus is suitable for many tissue papers as the short and slender fibres gives a high number of fibres per gram and low coarseness contributes to softness.
####Eucalyptus oil

Eucalyptus oil is readily steam distilled from the leaves and can be used for cleaning and as an industrial solvent, as an antiseptic, for deodorising, and in very small quantities in food supplements, especially sweets, cough drops, toothpaste and decongestants. It has insect-repellent properties, and serves as an active ingredient in some commercial mosquito-repellents. Aromatherapists have adopted Eucalyptus oils for a wide range of purposes. Eucalyptus globulus is the principal source of Eucalyptus oil worldwide.
####Musical instruments

Eucalypt wood is also commonly used to make didgeridoos, a traditional Australian Aboriginal wind instrument. The trunk of the tree is hollowed out by termites, and then cut down if the bore is of the correct size and shape.

Eucalypt wood is also being used as a tonewood and a fingerboard material for acoustic guitars, notably by the California-based Taylor company.
####Dyes

All parts of Eucalyptus may be used to make dyes that are substantive on protein fibres (such as silk and wool), simply by processing the plant part with water. Colours to be achieved range from yellow and orange through green, tan, chocolate and deep rust red. The material remaining after processing can be safely used as mulch or fertiliser.
####Prospecting

Eucalyptus trees in the Australian outback draw up gold from tens of metres underground through their root system and deposit it as particles in their leaves and branches. A Maia detector for x-ray elemental imaging at the Australian Synchrotron clearly showed deposits of gold and other metals in the structure of Eucalyptus leaves from the Kalgoorlie region of Western Australia that would have been untraceable using other methods. The microscopic leaf-bound "nuggets", about 8 micrometres wide on average, are not worth collecting themselves, but may provide an environmentally benign way of locating subsurface mineral deposits.
###Eucalyptus as plantation species

In the 20th century, scientists around the world experimented with Eucalyptus species. They hoped to grow them in the tropics, but most experimental results failed until breakthroughs in the 1960s-1980s in species selection, silviculture, and breeding programs "unlocked" the potential of eucalypts in the tropics. Prior to then, as Brett Bennett noted in a 2010 article, eucalypts were something of the "El Dorado" of forestry. Today, Eucalyptus is the most widely planted type of tree in plantations around the world, in South America (mainly in Brazil, Argentina, Paraguay and Uruguay), South Africa, Australia, India, Galicia, Portugal and many more.
####North America

#####California

In the 1850s, Eucalyptus trees were introduced to California by Australians during the California Gold Rush. Much of California is similar in climate to parts of Australia. By the early 1900s, thousands of acres of eucalypts were planted with the encouragement of the state government. It was hoped that they would provide a renewable source of timber for construction, furniture making and railway sleepers. It was soon found that for the latter purpose Eucalyptus was particularly unsuitable, as the ties made from Eucalyptus had a tendency to twist while drying, and the dried ties were so tough that it was nearly impossible to hammer rail spikes into them.

The species Eucalyptus rostrata, E. tereticornas, and E. cladocalyx are all present in California, but the blue gum E. globulus makes up by far the largest population in the state. One way in which the Eucalyptus, mainly the blue gum E. globulus, proved valuable in California was in providing windbreaks for highways, orange groves, and farms in the mostly treeless central part of the state. They are also admired as shade and ornamental trees in many cities and gardens.

Eucalyptus plantations in California have been criticised, because they compete with native plants and typically do not support native animals. Eucalyptus has historically been planted to replace California's coast live oak population, and the new Eucalyptus is not as hospitable to native flora and fauna as the oaks. In appropriately foggy conditions on the California Coast, Eucalyptus can spread at a rapid rate. The absence of natural inhibitors such as the koala or pathogens native to Australia have aided in the spread of California Eucalyptus trees. This is not as big of an issue further inland, but on the coast invasive eucalypts can disrupt native ecosystems. Eucalyptus may have adverse effects on local streams due to their chemical composition, and their dominance threatens species that rely on native trees. Nevertheless, some native species have been known to adapt to the Eucalyptus trees. Notable examples are herons, great horned owl, and the monarch butterfly using Eucalyptus groves as habitat. Despite these successes, eucalypts generally has a net negative impact on the overall balance of the native ecosystem.

Fire is also a problem. Eucalypts has been noted for their flammable properties and the large fuel loads in the understory of eucalypt forests. Eucalyptus trees were a catalyst for the spread of the 1923 fire in Berkeley, which destroyed 568 homes. The 1991 Oakland Hills firestorm, which caused US$1.5 billion in damage, destroyed almost 3,000 homes, and killed 25 people, was partly fueled by large numbers of eucalypts close to the houses.

Despite these issues, there are calls to preserve the Eucalyptus plants in California. Advocates for the tree claim its fire risk has been overstated. Some even claim that the Eucalyptus's absorption of moisture makes it a barrier against fire. These experts believe that the herbicides used to remove the Eucalyptus would negatively impact the ecosystem, and the loss of the trees would release carbon into the atmosphere unnecessarily. There is also an aesthetic argument for keeping the Eucalyptus; the trees are viewed by many as an attractive and iconic part of the California landscape. Many say that although the tree is not native, it has been in California long enough to become an essential part of the ecosystem and therefore should not be attacked as invasive. These arguments have caused experts and citizens in California, especially in the San Francisco Bay Area, to debate the merits of Eucalyptus removal versus preservation. However, the general consensus remains that some areas urgently require Eucalyptus management to stave off potential fire hazards.

Efforts to remove some of California's Eucalyptus trees have been met with a mixed reaction from the public, and there have been protests against removal. Removing Eucalyptus trees can be expensive and often requires machinery or the use of herbicides. The trees struggle to reproduce on their own outside of the foggy regions of Coastal California, and therefore some inland Eucalyptus forests are predicted to die out naturally. In some parts of California, eucalypt plantations are being removed and native trees and plants restored. Individuals have also illegally destroyed some trees and are suspected of introducing insect pests from Australia which attack the trees.

Certain Eucalyptus species may also be grown for ornament in warmer parts of the Pacific Northwest—western Washington, western Oregon and southwestern British Columbia.
####South America

#####Argentina

It was introduced in Argentina around 1870 by President Domingo F. Sarmiento, who had brought the seeds from Australia and it quickly became very popular. The most widely planted species were E. globulus, E. viminalis and E. rostrata. Currently, the Humid Pampas region has small forests and Eucalyptus barriers, some up to 80 years old, about 50 meters high and a maximum of one meter in diameter.

#####Uruguay

Antonio Lussich introduced Eucalyptus into Uruguay in approximately 1896, throughout what is now Maldonado Department, and it has spread all over the south-eastern and eastern coast. There had been no trees in the area because it consisted of dry sand dunes and stones. Lussich also introduced many other trees, particularly Acacia and pines, but they have not expanded so extensively.

Uruguayan forestry crops using Eucalyptus species have been promoted since 1989, when the new National Forestry Law established that 20% of the national territory would be dedicated to forestry. As the main landscape of Uruguay is grassland (140,000 km2, 87% of the national territory), most of the forestry plantations would be established in prairie regions. The planting of Eucalyptus sp. has been criticised because of concerns that soil would be degraded by nutrient depletion and other biological changes. During the last ten years, in the northwestern regions of Uruguay the Eucalyptus sp. plantations have reached annual forestation rates of 300%. That zone has a potential forested area of 1 million hectares, approximately 29% of the national territory dedicated to forestry, of which approximately 800,000 hectares are currently forested by monoculture of Eucalyptus spp. It is expected that the radical and durable substitution of vegetation cover leads to changes in the quantity and quality of soil organic matter. Such changes may also influence soil fertility and soil physical and chemical properties. The soil quality effects associated with Eucalyptus sp. plantations could have adverse effects on soil chemistry; for example: soil acidification, iron leaching, allelopathic activities and a high C:N ratio of litter. Additionally, as most scientific understanding of land cover change effects is related to ecosystems where forests were replaced by grasslands or crops, or grassland was replaced by crops, the environmental effects of the current Uruguayan land cover changes are not well understood. The first scientific publication on soil studies in western zone tree plantations (focused on pulp production) appeared in 2004 and described soil acidification and soil carbon changes, similar to a podzolisation process, and destruction of clay (illite-like minerals), which is the main reservoir of potassium in the soil. Although these studies were carried out in an important zone for forest cultivation, they cannot define the current situation in the rest of the land area under eucalyptus cultivation. Moreover, recently Jackson and Jobbagy have proposed another adverse environmental impact that may result from Eucalyptus culture on prairie soils—stream acidification.

The Eucalyptus species most planted are E. grandis, E. globulus and E. dunnii; they are used mainly for pulp mills. Approximately 80,000 ha of E. grandis situated in the departments of Rivera, Tacuarembó and Paysandú is primarily earmarked for the solid wood market, although a portion of it is used for sawlogs and plywood. The current area under commercial forest plantation is 6% of the total. The main uses of the wood produced are elemental chlorine free pulp mill production (for cellulose and paper), sawlogs, plywood and bioenergy (thermoelectric generation). Most of the products obtained from sawmills and pulp mills, as well as plywood and logs, are exported. This has raised the income of this sector with respect to traditional products from other sectors. Uruguayan forestry plantations have rates of growth of 30 cubic metres per hectare per year and commercial harvesting occurs after nine years.

#####Brazil

Eucalypts were introduced to Brazil in 1910, for timber substitution and the charcoal industry. It has thrived in the local environment, and today there are around 7 million hectares planted. The wood is highly valued by the charcoal and pulp and paper industries. The short rotation allows a larger wood production and supplies wood for several other activities, helping to preserve the native forests from logging. When well managed, the plantation soils can sustain endless replanting. Eucalyptus plantings are also used as wind breaks. Brazil's plantations have world-record rates of growth, typically over 40 cubic metres per hectare per year, and commercial harvesting occurs after years 5. Due to continual development and governmental funding, year-on-year growth is consistently being improved. Eucalyptus can produce up to 100 cubic metres per hectare per year. Brazil has become the top exporter and producer of Eucalyptus round wood and pulp, and has played an important role in developing the Australian market through the country's[clarification needed] committed research in this area. The local iron producers in Brazil rely heavily on sustainably grown Eucalyptus for charcoal; this has greatly pushed up the price of charcoal in recent years. The plantations are generally owned and operated for national and international industry by timber asset companies such as Thomson Forestry, Greenwood Management or cellulose producers such as Aracruz Cellulose and Stora Enso.

Overall, South America was expected to produce 55% of the world's Eucalyptus round-wood by 2010. Many environmental NGOs have criticised the use of exotic tree species for forestry in Latin America.
####Africa

#####Angola

In the East of Angola, the Benguela railway company created eucalyptus plantations for firing its steam locomotives.

#####Ethiopia

Eucalypts were introduced to Ethiopia in either 1894 or 1895, either by Emperor Menelik II's French advisor Mondon-Vidailhet or by the Englishman Captain O'Brian. Menelik II endorsed its planting around his new capital city of Addis Ababa because of the massive deforestation around the city for firewood. According to Richard R.K. Pankhurst, "The great advantage of the eucalypts was that they were fast growing, required little attention and when cut down grew up again from the roots; it could be harvested every ten years. The tree proved successful from the onset". Plantations of eucalypts spread from the capital to other growing urban centres such as Debre Marqos. Pankhurst reports that the most common species found in Addis Ababa in the mid-1960s was E. globulus, although he also found E. melliodora and E. rostrata in significant numbers. David Buxton, writing of central Ethiopia in the mid-1940s, observed that eucalyptus trees "have become an integral -- and a pleasing -- element in the Shoan landscape and has largely displaced the slow-growing native 'cedar' (Juniperus procera)."

It was commonly believed that the thirst of the Eucalyptus "tended to dry up rivers and wells", creating such opposition to the species that in 1913 a proclamation was issued ordering a partial destruction of all standing trees, and their replacement with mulberry trees. Pankhurst reports, "The proclamation however remained a dead letter; there is no evidence of eucalypts being uprooted, still less of mulberry trees being planted." Eucalypts remain a defining feature of Addis Ababa.

#####Madagascar

Much of Madagascar's original native forest has been replaced with Eucalyptus, threatening biodiversity by isolating remaining natural areas such as Andasibe-Mantadia National Park.

#####South Africa

Numerous Eucalyptus species have been introduced into South Africa, mainly for timber and firewood but also for ornamental purposes. They are popular with beekeepers for the honey they provide. However, in South Africa they are considered invasive, with their water-sucking capabilities threatening water supplies. They also release a chemical into the surrounding soil which kills native competitors.

Eucalyptus seedlings are usually unable to compete with the indigenous grasses, but after a fire when the grass cover has been removed, a seed-bed may be created. The following Eucalyptus species have been able to become naturalised in South Africa: E. camaldulensis, E. cladocalyx, E. diversicolor, E. grandis and E. lehmannii.

#####Zimbabwe

As in South Africa, many Eucalyptus species have been introduced into Zimbabwe, mainly for timber and firewood, and E. robusta and E. tereticornis have been recorded as having become naturalised there.
####Europe
#####Portugal

Eucalypts have been grown in Portugal since the mid 19th century, the first thought to be a specimen of E. obliqua introduced to Vila Nova de Gaia in 1829. First as an ornamental but soon after in plantations, these eucalypts are prized due to their long and upright trunks, rapid growth and the ability to regrow after cutting. These plantations now occupy around 800,000 hectares, 10% of the country's total land area, 90% of the trees being E. globulus. As of the late 20th century, there were an estimated 120 species of Eucalyptus in Portugal. The genus has also been subject to various controversies. Despite representing a large part of the agricultural economy, eucalypt plantations have a negative impact on soil destruction, inducing resistance to water infiltration and increasing the risks of erosion and soil loss, they are highly inflammable, aggravating the risk for wildfires. Various Portuguese laws on eucalypt plantations have been formed and reformed to better suit both sides.

There are various Eucalyptus species of public interest in Portugal, namely a Karri in Coimbra's Mata Nacional de Vale de Canas, considered to be Europe's tallest tree at 72 m (236 ft) high.
#####Italy

In Italy, the Eucalyptus only arrived at the turn of the 19th century and large scale plantations were started at the beginning of the 20th century with the aim of drying up swampy ground to defeat malaria. During the 1930s, Benito Mussolini had thousands of Eucalyptus planted in the marshes around Rome. This, their rapid growth in the Italian climate and excellent function as windbreaks, has made them a common sight in the south of the country, including the islands of Sardinia and Sicily. They are also valued for the characteristic smelling and tasting honey that is produced from them. The variety of Eucalyptus most commonly found in Italy is E. camaldulensis.
#####Greece

In Greece, eucalypts are widely found, especially in southern Greece and Crete. They are cultivated and used for various purposes, including as an ingredient in pharmaceutical products (e.g., creams, elixirs and sprays) and for leather production. They were imported in 1862 by botanist Theodoros Georgios Orphanides. The principal species is E. globulus.
#####Ireland

Eucalyptus has been grown in Ireland since trials in the 1930s and now grows wild in South Western Ireland in the mild climate.
####Asia

Eucalyptus seeds of the species E. globulus were imported into Palestine in the 1860s, but did not acclimatise well. Later, E. camaldulensis was introduced more successfully and it is still a very common tree in Israel. The use of Eucalyptus trees to drain swampy land was a common practice in the late nineteenth and early twentieth centuries. The German Templer colony of Sarona had begun planting Eucalyptus for this purpose by 1874, though it is not known where the seeds came from. Many Zionist colonies also adopted the practice in the following years under the guidance of the Mikveh Israel Agricultural School. Eucalyptus trees are now considered an invasive species in the region.

In India, the Institute of Forest Genetics and Tree Breeding, Coimbatore started a Eucalyptus breeding program in the 1990s. The organisation released four varieties of conventionally bred, high yielding and genetically improved clones for commercial and research interests in 2010.

Eucalyptus trees were introduced to Sri Lanka in the late 19th century by tea and coffee planters, for wind protection, shade and fuel. Forestry replanting of Eucalyptus began in the 1930s in deforested mountain areas, and currently there are about 10 species present in the island. They account for 20% of major reforestation plantings. They provide railway sleepers, utility poles, sawn timber and fuelwood, but are controversial because of their adverse effect on biodiversity, hydrology and soil fertility. They are associated with another invasive species, the eucalyptus gall wasp, Leptocybe invasa.
####Pacific Islands

In Hawaii, some 90 species of Eucalyptus have been introduced to the islands, where they have displaced some native species due to their higher maximum height, fast growth and lower water needs. Particularly noticeable is the rainbow eucalyptus (Eucalyptus deglupta), native to Indonesia and the Philippines, whose bark falls off to reveal a trunk that can be green, red, orange, yellow, pink and purple.
###Non-native Eucalyptus and biodiversity

Due to similar favourable climatic conditions, Eucalyptus plantations have often replaced oak woodlands, for example in California, Spain and Portugal. The resulting monocultures have raised concerns about loss of biological diversity, through loss of acorns that mammals and birds feed on, absence of hollows that in oak trees provide shelter and nesting sites for birds and small mammals and for bee colonies, as well as lack of downed trees in managed plantations. A study of the relationship between birds and Eucalyptus in the San Francisco Bay Area found that bird diversity was similar in native forest versus Eucalyptus forest, but the species were different. One way in which the avifauna (local assortment of bird species) changes is that cavity-nesting birds including woodpeckers, owls, chickadees, wood ducks, etc. are depauperate in Eucalyptus groves because the decay-resistant wood of these trees prevents cavity formation by decay or excavation. Also, those bird species that glean insects from foliage, such as warblers and vireos, experience population declines when Eucalyptus groves replace oak forest.

Birds that thrive in Eucalyptus groves in California tend to prefer tall vertical habitat. These avian species include herons and egrets, which also nest in redwoods. The Point Reyes Bird Observatory observes that sometimes short-billed birds like the ruby-crowned kinglet are found dead beneath Eucalyptus trees with their nostrils clogged with pitch.

Monarch butterflies use Eucalyptus in California for overwintering, but in some locations have a preference for Monterey pines.
####Eucalyptus as an invasive species

Eucalyptus trees are considered invasive to local ecosystems and negatively impact water resources in countries where they are introduced.

#####South Africa

In South Africa, Eucalyptus tree species E. camaldulensis, E. cladocalyx, E. conferruminata, E. diversicolor, E. grandis and E. tereticornis are listed as Category 1b invaders in the National Environmental Management: Biodiversity Act. This means most activities with regards to the species are prohibited (such as importing, propagating, translocating or trading) and it should be ensured that it does not spread beyond a plantation's domain.

E. cladocalyx and E. diversicolor are considered Fynbos invaders, and use up to 20% more water than the native fynbos vegetation; with invasive species including Eucalyptus being cleared that reduce Cape Town's water resource by 55 billion litres or two months worth of water supply. 
'''),
    SpeciesDetail('Euxylophora paraensis', '''
Euxylophora paraensis is a species of tree in the family Rutaceae. They have a self-supporting growth form. They are listed as endangered by IUCN. They are native to Amazônia and South America. They have compound, broad leaves.
''', "https://eol.org/pages/5624064", '''
'''),
    SpeciesDetail('Goupia glabra', '''
Goupia glabra (goupie or kabukalli; syn. G. paraensis, G. tomentosa) is a species of flowering plant in the family Goupiaceae (formerly treated in the family Celastraceae). It is native to tropical South America, in northern Brazil, Colombia, French Guiana, Guyana, Suriname, and Venezuela.
Other names include Saino, Sapino (Colombia), Kopi (Surinam), Kabukalli (Guyana), Goupi, bois-caca (French Guiana), Pasisi (Wayampi language), Pasis (Palikur language), Kopi (Businenge language), Cupiuba (Brazil), yãpi mamo hi (Yanomami language), Venezuela. 
''', "https://en.wikipedia.org/wiki/Goupia_glabra", '''
Goupia glabra (goupie or kabukalli; syn. G. paraensis, G. tomentosa) is a species of flowering plant in the family Goupiaceae (formerly treated in the family Celastraceae). It is native to tropical South America, in northern Brazil, Colombia, French Guiana, Guyana, Suriname, and Venezuela.

Other names include Saino, Sapino (Colombia), Kopi (Surinam), Kabukalli (Guyana), Goupi, bois-caca (French Guiana), Pasisi (Wayampi language), Pasis (Palikur language), Kopi (Businenge language), Cupiuba (Brazil), yãpi mamo hi (Yanomami language), Venezuela.
###Description

It is a large, fast-growing tree growing up to 50 m tall with a trunk up to 1.3 m diameter, often buttressed at the base up to 2 m diameter, with rough, silvery-grey to reddish-grey bark. It is usually evergreen, but can be deciduous in the dry season. The leaves are alternate, broad lanceolate, with an entire margin and a petiole with a complex vascular system. The flowers are small, yellow-green, with five sepals and petals; they are produced in clusters, and are wind-pollinated. The fruit is an orange-red berry-like drupe 5 mm diameter, containing 5–10 seeds; it is eaten by various birds (including cotingas, pigeons, tanagers, thrushes, and trogons), which disperse the seeds in their droppings. 
'''),
    SpeciesDetail('Grevilea robusta', '''
Grevillea robusta, commonly known as the southern silky oak, silk oak or silky oak, silver oak or Australian silver oak, is a flowering plant in the family Proteaceae, and accordingly unrelated to true oaks, family Fagaceae. Grevillea robusta is a tree, and is the largest species in its genus. It is a native of eastern coastal Australia, growing in riverine, subtropical and dry rainforest environments. 
''', "https://en.wikipedia.org/wiki/Grevillea_robusta", '''
Grevillea robusta, commonly known as the southern silky oak, silk oak or silky oak, silver oak or Australian silver oak, is a flowering plant in the family Proteaceae, and accordingly unrelated to true oaks, family Fagaceae. Grevillea robusta is a tree, and is the largest species in its genus. It is a native of eastern coastal Australia, growing in riverine, subtropical and dry rainforest environments.
###Description

Grevillea robusta is a fast-growing evergreen tree with a single main trunk, growing to 5–40 m (20–100 ft) tall. The bark is dark grey and furrowed. Its leaves are fern-like, 10–34 cm (4–10 in) long, 9–15 cm (4–6 in) wide and divided with between 11 and 31 main lobes. Each lobe is sometimes further divided into as many as four, each one linear to narrow triangular in shape. It loses many of its leaves just before flowering.

The flowers are arranged in one-sided, "toothbrush"-like groups, sometimes branched, 12–15 cm (5–6 in) long. The carpel (the female part) of each flower has a stalk 21–28 mm (0.8–1 in) long. The flowers are glabrous and mostly yellowish orange, or sometimes reddish. Flowering occurs from September to November and the fruit that follows is a glabrous follicle.
###Taxonomy and naming

Grevillea robusta was first formally described in 1830 by Robert Brown after an unpublished description by Allan Cunningham. The type specimen was collected by Cunningham on the eastern edge of Moreton Bay in 1827. Brown's description was published in Supplementum primum Prodromi florae Novae Hollandiae. The specific epithet (robusta) is a Latin word meaning "strong like oak" or "robust".
###Distribution and habitat

Silky oak occurs naturally on the coast and ranges in southern Queensland and in New South Wales as far south as Coffs Harbour where it grows in subtropical rainforest, dry rainforest and wet forests. It is now relatively rare in its natural habitat but has been widely planted, including on Norfolk Island and Lord Howe Island. It has become naturalised in many places, including on the Atherton Tableland in Australia and in South Africa, New Zealand, Hawaii, French Polynesia, Jamaica and Florida. It is regarded as a weed in parts of New South Wales and Victoria, as "invasive" in Hawaii and as an "invader" in South Africa.
###Uses

Before the advent of aluminium, Grevillea robusta timber was widely used for external window joinery, as it is resistant to wood rot. It has been used in the manufacture of furniture, cabinetry, and fences. Owing to declining G. robusta populations, felling has been restricted.

Recently G. robusta has been used for side and back woods on guitars made by Larrivée and others, because of its tonal and aesthetic qualities.
###Cultivation

When young, it can be grown as a houseplant where it can tolerate light shade, but it prefers full sun because it grows best in warm zones. If planted outside, young trees need protection on frosty nights. Once established it is hardier and tolerates temperatures down to −8 °C (18 °F). It needs occasional water but is otherwise fairly drought-resistant. Care needs to be taken when planting it near bushland because it can be invasive.

G. robusta is often used as stock for grafting difficult-to-grow grevilleas. It has been planted widely throughout the city of Kunming in south-western China, forming shady avenues.

G. robusta is grown in plantations in South Africa, and can also be grown alongside maize in agroforestry systems.

In the UK, G. robusta has gained the Royal Horticultural Society's Award of Garden Merit.
###Toxicity and allergic reactions

The flowers and fruit contain toxic hydrogen cyanide. Tridecylresorcinol in G.robusta is responsible for contact dermatitis. 
'''),
    SpeciesDetail('Hura crepitans', '''
Hura crepitans, the sandbox tree, also known as possumwood, monkey no-climb, assacu (from Tupi asaku) and jabillo, is an evergreen tree in the family Euphorbiaceae, native to tropical regions of North and South America including the Amazon rainforest. It is also present in parts of Tanzania, where it is considered an invasive species. Because its fruits explode when ripe, it has also received the colloquial nickname the dynamite tree.
''', "https://en.wikipedia.org/wiki/Hura_crepitans", '''
Hura crepitans, the sandbox tree, also known as possumwood, monkey no-climb, assacu (from Tupi asaku) and jabillo, is an evergreen tree in the family Euphorbiaceae, native to tropical regions of North and South America including the Amazon rainforest. It is also present in parts of Tanzania, where it is considered an invasive species. Because its fruits explode when ripe, it has also received the colloquial nickname the dynamite tree.
###Description

The sandbox tree can grow to 60 metres (200 ft), and its large ovate leaves grow to 60 cm (2 ft) wide. They are monoecious, with red, un-petaled flowers. Male flowers grow on long spikes, while female flowers grow alone in leaf axils. The trunk is covered in long, sharp spikes that secrete poisonous sap. The sandbox tree's fruits are large, pumpkin-shaped capsules, 3–5 cm (1–2 in) long, 5–8 cm (2–3 in) diameter, with 16 carpels arranged radially. Its seeds are flattened and about 2 cm (3⁄4 in) diameter. The capsules explode when ripe, splitting into segments and launching seeds at 70 m/s (250 km/h; 160 mph). One source states that ripe capsules catapult their seeds as far as 100 m (330 ft). Another source states that seeds are thrown as far as 45 m (150 ft) from a tree, averaging about 30 m (100 ft).
###Habitat

This tree prefers wet soil, and partial shade or partial to full sun. It is often cultivated for shade. Sandbox trees are tropical trees and prefer warmer, more humid environments.
###Uses

Its wood is light enough to make indigenous canoes. Fishermen have been said to use the milky, caustic sap from this tree to poison fish. The Caribs made arrow poison from its sap. The wood is used for furniture under the name "hura". Before more modern forms of pens were invented, the trees' unripe seed capsules were sawn in half to make decorative pen sandboxes (also called pounce pots), hence the name 'sandbox tree'. It has been documented as a herbal remedy.

The seeds contain an oil that is toxic for consumption but can be made into biodiesel and soap, its starchy leftovers can be made into animal feed after cooking. 
'''),
    SpeciesDetail('Hymenaea sp', '''
Hymenaea is a genus of plants in the legume family Fabaceae. Of the fourteen living species in the genus, all but one are native to the tropics of the Americas, with one additional species (Hymenaea verrucosa) on the east coast of Africa. Some authors place the African species in a separate monotypic genus, Trachylobium. In the Neotropics, Hymenaea is distributed through the Caribbean islands, and from southern Mexico to Brazil. Linnaeus named the genus in 1753 in Species Plantarum for Hymenaios, the Greek god of marriage ceremonies. The name is a reference to the paired leaflets. 
''', "https://en.wikipedia.org/wiki/Hymenaea", '''
Hymenaea is a genus of plants in the legume family Fabaceae. Of the fourteen living species in the genus, all but one are native to the tropics of the Americas, with one additional species (Hymenaea verrucosa) on the east coast of Africa. Some authors place the African species in a separate monotypic genus, Trachylobium. In the Neotropics, Hymenaea is distributed through the Caribbean islands, and from southern Mexico to Brazil. Linnaeus named the genus in 1753 in Species Plantarum for Hymenaios, the Greek god of marriage ceremonies. The name is a reference to the paired leaflets.

Most species of Hymenaea are large trees and they are primarily evergreen. They may grow to a height of 25 m (82 ft) and emerge above the forest canopy. Some species will grow both as tall forest trees and as smaller shrubby trees depending on their surrounding habitat. The leaves are pinnately bifoliolate, meaning that they have two leaflets attached to the sides of the petiole. The flowers grow in a panicle or corymb type of inflorescence.
###Uses and properties

The pulpy center of the fruits is edible and contains starch. The fruit is sold in local markets in the Americas. The leaves may be used to make a tea. The trees produce a dense wood used for timber in making ships and furniture. The thick bark of some species is used by indigenous peoples of the Amazon to make canoes. Seeds contain large amounts (40% of dry weight) of a highly viscous polysaccharide (xyloglucan) which can be used in several industrial sectors such as food, paper, cosmetic and pharmaceutical.

The trees also make hard resins that are used to manufacture varnish, especially the resin from Hymenaea courbaril (jatobá) in Brazil. The resin that is produced in Brazil is known as South American copal, and Hymenaea verrucosa is the source of the valuable Zanzibar copal. Resin may be collected from living trees, or from the soil near the place where a tree once stood. Throughout its American range, indigenous peoples use the resin for incense and as a cement. Resin from the extinct species Hymenaea protera is the source of Dominican amber, while the extinct Hymenaea mexicana produced the resin which is the source of Mexican amber.

Hymenaea courbaril has been used as a model organism to study the effect of increased CO2 concentration on the rate of photosynthesis in neotropical regions.: 10 

When the concentration of CO2 was increased from an ambient reference level of 360ppm to 720ppm, the photosynthetic CO2 assimilation in the seedlings doubled.: 3  This suggests the species could play an important role in greenhouse gas sequestration, as atmospheric CO2 is expected to reach ca. 700 ppm by the year 2075 if current levels of fossil fuel consumption are maintained.

Hymenaea courbaril is a very important species in programmes of recuperation of degraded rain forests in the Neotropics. It appears late in the natural regeneration process being classified as a 'late successional' or climax species. 
'''),
    SpeciesDetail('Hymenolobium petraeum', '''
Hymenolobium petraeum is a species of tree in the family legumes. They have a self-supporting growth form. They are native to Amazônia, Amapá, Pará, and Maranhao. They have compound, broad leaves.
''', "https://eol.org/pages/417255", '''
'''),
    SpeciesDetail('Laurus nobilis', '''
Laurus nobilis /ˈlɔːrəs ˈnɒbɪlɪs/ is an aromatic evergreen tree or large shrub with green, glabrous (smooth) leaves. It is in the flowering plant family Lauraceae. It is native to the Mediterranean region and is used as bay leaf for seasoning in cooking. Its common names include bay tree (esp. United Kingdom),: 84  bay laurel, sweet bay, true laurel, Grecian laurel, or simply laurel. Laurus nobilis figures prominently in classical Greco-Roman culture.
Worldwide, many other kinds of plants in diverse families are also called "bay" or "laurel", generally due to similarity of foliage or aroma to Laurus nobilis. 
''', "https://en.wikipedia.org/wiki/Laurus_nobilis", '''
Laurus nobilis /ˈlɔːrəs ˈnɒbɪlɪs/ is an aromatic evergreen tree or large shrub with green, glabrous (smooth) leaves. It is in the flowering plant family Lauraceae. It is native to the Mediterranean region and is used as bay leaf for seasoning in cooking. Its common names include bay tree (esp. United Kingdom),: 84  bay laurel, sweet bay, true laurel, Grecian laurel, or simply laurel. Laurus nobilis figures prominently in classical Greco-Roman culture.

Worldwide, many other kinds of plants in diverse families are also called "bay" or "laurel", generally due to similarity of foliage or aroma to Laurus nobilis.
###Description

The laurel is an evergreen shrub or small tree, variable in size and sometimes reaching 7–18 m (23–59 ft) tall. The genus Laurus includes three accepted species, whose diagnostic key characters often overlap.

The bay laurel is dioecious (unisexual), with male and female flowers on separate plants. Each flower is pale yellow-green, about 1 cm (3⁄8 in) diameter, and they are borne in pairs beside a leaf. The leaves are glabrous, 6–12 cm (2–5 in) long and 2–4 cm (3⁄4–1+5⁄8 in) broad, with an entire (untoothed) margin. On some leaves the margin undulates. The fruit is a small, shiny black drupe-like berry about 1 cm (3⁄8 in) long that contains one seed.
###Ecology

Laurus nobilis is a widespread relict of the laurel forests that originally covered much of the Mediterranean Basin when the climate of the region was more humid. With the drying of the Mediterranean during the Pliocene era, the laurel forests gradually retreated, and were replaced by the more drought-tolerant sclerophyll plant communities familiar today. Most of the last remaining laurel forests around the Mediterranean are believed to have disappeared approximately ten thousand years ago, although some remnants still persist in the mountains of southern Turkey, northern Syria, southern Spain, north-central Portugal, northern Morocco, the Canary Islands and in Madeira.
###Human uses
####Food

The plant is the source of several popular herbs and one spice used in a wide variety of recipes, particularly among Mediterranean cuisines. Most commonly, the aromatic leaves are added whole to Italian pasta sauces. They are typically removed from dishes before serving, unless used as a simple garnish. Whole bay leaves have a long shelf life of about one year, under normal temperature and humidity. Whole bay leaves are used almost exclusively as flavor agents during the food preparation stage.

Ground bay leaves, however, can be ingested safely and are often used in soups and stocks, as well as being a common addition to a Bloody Mary. Dried laurel berries and pressed leaf oil can both be used as robust spices, and the wood can be burnt for strong smoke flavoring.
####Ornamental

Laurus nobilis is widely cultivated as an ornamental plant in regions with Mediterranean or oceanic climates, and as a house plant or greenhouse plant in colder regions. It is used in topiary to create single erect stems with ball-shaped, box-shaped or twisted crowns; also for low hedges. However it is slow-growing and may take several years to reach the desired height. Together with a gold form, L. nobilis 'Aurea'  and a willow-leaved form L. nobilis f. angustifolia, it has gained the Royal Horticultural Society's Award of Garden Merit.

One of the most important pests affecting ornamental laurels is caused by the jumping plant louse Trioza alacris, which induces the curling and thickening of the edge of the leaves for the development of the insect's nymphs, eventually creating a necrosed gall. The species is also affected by the scale insect Coccus hesperidum.
####Alternative medicine

In herbal medicine, aqueous extracts of bay laurel have been used as an astringent and salve for open wounds. It is also used in massage therapy and aromatherapy. A folk remedy for rashes caused by poison ivy, poison oak, and stinging nettle is a poultice soaked in boiled bay leaves. The Roman naturalist Pliny the Elder listed a variety of conditions which laurel oil was supposed to treat: paralysis, spasms, sciatica, bruises, headaches, catarrhs, ear infections, and rheumatism.
####Other uses

Laurel oil is a secondary ingredient, and the distinguishing fragrant characteristic of Aleppo soap.
###Symbolism
####Greece

In Greek, the plant is called δάφνη : dáphnē, after the mythic mountain nymph of the same name. In the myth of Apollo and Daphne, the god Apollo fell in love with Daphne, a priestess of Gaia (Mother Earth), and when he tried to seduce her she pleaded for help to Gaia, who transported her to Crete. In Daphne's place Gaia left a laurel tree, from which Apollo fashioned wreaths to console himself.

Other versions of the myth, including that of the Roman poet Ovid, state that Daphne was transformed directly into a laurel tree.

Bay laurel was used to fashion the laurel wreath of ancient Greece, a symbol of highest status. A wreath of bay laurels was given as the prize at the Pythian Games because the games were in honor of Apollo, and the laurel was one of his symbols. According to the poet Lucian, the priestess of Apollo known as the Pythia reputedly chewed laurel leaves from a sacred tree growing inside the temple to induce the enthusiasmos (trance) from which she uttered the oracular prophecies for which she was famous. Some accounts starting in the fourth century BC describe her as shaking a laurel branch while delivering her prophecies. Those who received promising omens from the Pythia were crowned with laurel wreaths as a symbol of Apollo's favor.
####Rome

The symbolism carried over to Roman culture, which held the laurel as a symbol of victory. It was also associated with immortality, with ritual purification, prosperity and health. It is also the source of the words baccalaureate and poet laureate, as well as the expressions "assume the laurel" and "resting on one's laurels".

Pliny the Elder stated that the laurel was not permitted for "profane" uses – lighting it on fire at altars "for the propitiation of divinities" was strictly forbidden, because "it is very evident that the laurel protests against such usage by crackling as it does in the fire, thus, in a manner, giving expression to its abhorrence of such treatment".

Laurel was closely associated with the Roman Emperors, beginning with Augustus. Two Laurel trees flanked the entrance to Augustus' house on the Palatine Hill in Rome, which itself was connected to the Temple of Apollo Palatinus, which Augustus had built. Thus, the laurels had the dual purpose of advertising Augustus' victory in the Civil Wars and his close association with Apollo. Suetonius relates the story of Augustus' wife, and Rome's first Empress, Livia, who planted a sprig of laurel on the grounds of her villa at Prima Porta after an eagle dropped a hen with the sprig clutched in its beak onto her lap. The sprig grew into a full-size tree which fostered an entire grove of laurel trees, which were in turn added to by subsequent Emperors when they celebrated a triumph. The Emperors in the Julio-Claudian dynasty all sourced their Laurel wreaths from the original tree planted by Livia. It was taken as an omen of the impending end of the Julio-Claudian dynasty that in the reign of Nero the entire grove died, shortly before he was assassinated. Rome's second Emperor Tiberius wore wreaths of laurel whenever there was stormy weather because it was widely believed that Laurel trees were immune to lightning strikes, affording protection to those who brandished it. One reason for this belief is because laurel crackles loudly when on fire. It led ancient Romans to believe the plant was inhabited by a "heavenly fire demon", and was therefore "immune" from outer threats like fire or lightning.

In modern Italy, laurel wreaths are worn as a crown by graduating school students.
####East Asia

An early Chinese etiological myth for the phases of the moon involved a great forest or tree which quickly grew and lost its leaves and flowers every month. After the Sui and Tang dynasties, this was sometimes connected to a woodsman named Wu Gang, sentenced to cut at a self-repairing tree as a punishment for varying offenses. The tree was originally identified as a 桂 (guì) and described in the terms of the osmanthus (Osmanthus fragrans, now known in Chinese as the 桂花 or "gui flower"), whose blossoms are still used to flavor wine and confections for the Mid-Autumn Festival. However, in English, it is often associated with the more well-known cassia (Cinnamomum cassia, now known in Chinese as the 肉桂 or "meat gui") while, in modern Chinese, it has instead become associated with the Mediterranean laurel. By the Qing dynasty, the chengyu "pluck osmanthus in the Toad Palace" (蟾宫折桂, Chángōng zhé guì) meant passing the imperial examinations, which were held around the time of the lunar festival. The similar association in Europe of laurels with victory and success led to its translation into Chinese as the 月桂 or "Moon gui".
####Finland

The laurel leaves in the coat of arms of Kaskinen, Finland (Swedish: Kaskö) may have been meant to refer to local flowering, but its origin may also be in the name of the family Bladh (Swedish: blad; 'leaf'); two members of the family – a father and a son – acquired both town rights and the status of staple town for the village at the time.
###Chemical constituents

The most abundant component found in laurel essential oil is 1,8-cineole, also called eucalyptol. The leaves contain about 1.3% essential oils (ol. lauri folii), consisting of 45% eucalyptol, 12% other terpenes, 8–12% terpinyl acetate, 3–4% sesquiterpenes, 3% methyleugenol, and other α- and β-pinenes, phellandrene, linalool, geraniol, and terpineol. It contains lauric acid also.

Both essential and fatty oils are present in the fruit. The fruit is pressed and water-extracted to obtain these products. The fruit contains up to 30% fatty oils and about 1% essential oils (terpenes, sesquiterpenes, alcohols, and ketones). This laurel oil is the characteristic ingredient of Aleppo soap. The chemical compound lauroside B has been isolated from Laurus nobilis. 
'''),
    SpeciesDetail('Machaerium sp', '''
Machaerium is a genus of flowering plants in the family Fabaceae, and was recently assigned to the informal monophyletic Dalbergia clade of the Dalbergieae.
''', "https://en.wikipedia.org/wiki/Machaerium_(plant)", '''
'''),
    SpeciesDetail('Manilkara huberi', '''
Manilkara huberi, also known as masaranduba, níspero, and sapotilla, is a fruit bearing plant of the genus Manilkara of the family Sapotaceae. 
''', "https://en.wikipedia.org/wiki/Manilkara_huberi", '''
Manilkara huberi, also known as masaranduba, níspero, and sapotilla, is a fruit bearing plant of the genus Manilkara of the family Sapotaceae.
###Geographical distribution

Manilkara huberi is native to large parts of northern South America, Central America and the Antilles, at elevations below 800 metres (2,600 ft) above sea level.
###Description

Manilkara huberi is a large tree, reaching heights of 30–55 metres (98–180 ft). The leaves are oblong, approximately 1–2 decimetres (3.9–7.9 in) in length, with yellow undersides. The flowers are hermaphroditic; white with 3 sepals. The edible fruit is yellow and ovoid, 3 centimetres (1.2 in) in diameter, containing one seed (or occasionally two).
###Uses

The fruit of the M. huberi is similar to the sapodilla and is edible, with excellent flavor popular for use in desserts.

M. huberi produces an edible latex that can be harvested in a manner similar to the harvesting of the latex of the rubber tree (Hevea brasiliensis). The latex dries to an inelastic rubber, which is considered inferior to gutta-percha.

The latex from M. huberi is sometimes used to make golf ball covers. It is considered a good, but short-lived, cover, requiring frequent recoating, yet it is popular in tournaments.

The tree is also used for lumber in Puerto Rico. The wood is red and very hard, and is popular for use in furniture making, construction, and railway ties. The wood is so dense to the point that it does not float on water, and requires pre-drilling before nailing. The specific gravity of M. huberi wood is between 0.85 and 0.95 g/cm3. 
'''),
    SpeciesDetail('Melia azedarach', '''
Melia azedarach, commonly known as the chinaberry tree, pride of India, bead-tree, Cape lilac, syringa berrytree, Persian lilac, Indian lilac, or white cedar, is a species of deciduous tree in the mahogany family, Meliaceae, that is native to Indomalaya and Australasia.
''', "https://en.wikipedia.org/wiki/Melia_azedarach", '''
Melia azedarach, commonly known as the chinaberry tree, pride of India, bead-tree, Cape lilac, syringa berrytree, Persian lilac, Indian lilac, or white cedar, is a species of deciduous tree in the mahogany family, Meliaceae, that is native to Indomalaya and Australasia.
###Description

The fully grown tree has a rounded crown, and commonly measures 7–12 metres (20–40 feet) tall, exceptionally 45 m (150 ft).

The leaves are up to 50 centimetres (20 inches) long, alternate, long-petioled, two or three times compound (odd-pinnate); the leaflets are dark green above and lighter green below, with serrate margins.

The flowers are small and fragrant, with five pale purple or lilac petals, growing in clusters.

The fruit is a drupe, marble-sized, light yellow at maturity, hanging on the tree all winter, and gradually becoming wrinkled and almost white.
####Chemistry

Italo et al. 2009 and Safithri and Sari 2016 report flavonoids and phenols found in M. azedarach.: 490 
###Etymology

The genus name Melia is derived from μελία (melía), the Greek word used by Theophrastus (c. 371 – c. 287 BC) for Fraxinus ornus, which has similar leaves. The species azedarach is from the French 'azédarac' which in turn is from the Persian 'āzād dirakht' (ازادرخت ) meaning 'free- or noble tree'.[full citation needed]

Melia azedarach should not be confused with the Azadirachta trees, which are in the same family, but a different genus.
###Ecology

Some hummingbirds like the sapphire-spangled emerald (Amazilia lactea), glittering-bellied emerald (Chlorostilbon lucidus) and planalto hermit (Phaethornis pretrei) have been recorded as feeding on and pollinating the flowers; these only take it opportunistically.[page needed]

Bees and butterflies do not use the flower (or the nectar) so it serves no pollinator benefit in the US.

Pests such as cape lilac tree caterpillars, Leptocneria reducta, can severely defoliate the tree and cause a lot of damage to the tree in Australia.

Fungal plant pathogen Pseudocercospora subsessilis is found on the leaves of the tree, causing leaf spots.

A mature Chinaberry tree is environment-versatile and can withstand temperatures as low as -5˚C and can survive in warm temperatures up to 39˚C. Although, according to the USDA, the tree exists as far up north as New York(Distribution data)
###Uses

The plant was introduced around 1830 as an ornamental in the United States (South Carolina and Georgia) and widely planted in southern states. It was introduced into Hawaii in 1840. It is considered an invasive species in Texas, and by some American groups as far north as Virginia and Oklahoma. But US nurseries continue to sell the trees, and the seeds are also widely available. It has become naturalized to tropical and warm temperate regions of the Americas and is planted in similar climates around the world. It is an ornamental tree in the southern part of Korea.

It was planted in parks, public gardens, stream banks and along footpaths or roadsides in Australia. The fragrant lilac flowers and yellow fruits of White Cedar make it an appealing ornamental tree. The hard seeds of the plant could also be used in art and crafts, such as making beads for rosaries. It has naturalized in parts of Australia and in New Zealand, but it is classed as 'weed', since it has the ability to colonise an area (with bird dropped seed) if left unchecked.
###Toxicity

The fruits have evolved to be eaten by animals, which eat the flesh surrounding the hard endocarp or ingest the entire fruit and later vent the endocarp. If the endocarp is crushed or damaged during ingestion or digestion, the animal will be exposed to the toxins within the seed. The processes of mastication and digestion, and the degree of immunity to the particular toxins, vary widely between species, and there will accordingly be great variation in the clinical symptoms following ingestion.

Fruits are poisonous or narcotic to humans if eaten in large quantities.[page needed] According to Chinese medical literature, human poisoning can occur if 6 - 9 fruits, 30 - 40 seeds, or 400 grams of the bark are eaten. However, these toxins are not harmful to birds, who gorge themselves on the fruit, eventually reaching a "drunken" state. The birds that are able to eat the fruit spread the seeds in their droppings. The toxins are neurotoxins and unidentified resins, found mainly in the fruits. The first symptoms of poisoning appear a few hours after ingestion. They may include loss of appetite, vomiting, constipation or diarrhea, bloody faeces, stomach pain, pulmonary congestion, cardiac arrest, rigidity, lack of coordination and general weakness. Death may take place after about 24 hours. As in relatives, tetranortriterpenoids constitute an important toxic principle. These are chemically related to azadirachtin, the primary insecticidal compound in the commercially important neem oil. These compounds are probably related to the wood and seed's resistance to pest infestation, and maybe to the unattractiveness of the flowers to animals.

The plant is toxic to cats.
###Uses

The main utility of chinaberry is its timber. This is of medium density, and ranges in colour from light brown to dark red. In appearance it is readily confused with the unrelated Burmese teak (Tectona grandis). Melia azedarach, in keeping with other members of the family Meliaceae, has a timber of high quality, but in comparison to many almost-extinct species of mahogany, it is under-utilised. Seasoning is relatively simple — planks dry without cracking or warping and are resistant to fungal infection.

The tough five-grooved seeds were widely used for making rosaries and other products requiring beads; however, the seeds were later replaced by plastics. The cut branches with mature fruit are sold commercially to the florist and landscaping trade particularly as a component for outdoor holiday décor. The fruits may persist for some time prior to shattering off the stem or discoloring, which occurs rapidly after a relatively short time in subfreezing weather.

In Kenya the trees have been grown by farmers and used as fodder trees. The leaves can be fed to cattle to improve milk yields and improve farm incomes. The taste of the leaves is not as bitter as that of the leaves of neem (Azadirachta indica).

In Australia, particularly the suburbs of Melbourne, the tree is often used in nature strip plantings by local councils. The councils plant such trees for amenity reasons as well as environmental, social and economic benefits.

Leaves have been used as a natural insecticide to keep with stored food, but must not be eaten as they are highly poisonous. Chinaberry fruit was used to prevent insect larvae from growing in the fruit. By placing the berries in, for example, drying apples and keeping the fruit turned in the sun without damaging any of the chinaberry skin, the fruit will dry and will prevent insect larvae in the dried apples. A mature tree can yield approximately 15 kilograms of fruit annually.

A diluted infusion of leaves and trees has been used in the past to induce uterine relaxation.

The tree's Limonoid contains useful anticancer and antimalarial compounds. 
'''),
    SpeciesDetail('Mezilaurus itauba', '''
Mezilaurus itauba is a species of tree in the family Lauraceae. It is found in Bolivia, Brazil, Ecuador, French Guiana, Peru, and Suriname.
''', "https://en.wikipedia.org/wiki/Mezilaurus_itauba", '''
'''),
    SpeciesDetail('Micropholis venulosa', '''
Micropholis venulosa is a species of tree in the family Sapotaceae. They have a self-supporting growth form. They are native to Bahia, Mato Grosso, Pará, Maranhao, Espirito Santo, GoiáS, Cerrado, Amazônia, Distrito Federal, Mata Atlântica, Minas Gerais, Acre (Brazil), Mato Grosso Do Sul, RondôNia, and The Neotropics. They have simple, broad leaves. Individuals can grow to 21 m.
''', "https://eol.org/pages/1154259", '''
'''),
    SpeciesDetail('Mimosa scabrella', '''
Mimosa scabrella is a tree in the family Fabaceae. It is very fast-growing and it can reach a height of 15 m (49 ft) tall in only 3 years. Its trunk is about 0.1–0.5 m (3.9–19.7 in) in diameter. It has yellow flowers.
''', "https://en.wikipedia.org/wiki/Mimosa_scabrella", '''
Mimosa scabrella is a tree in the family Fabaceae. It is very fast-growing and it can reach a height of 15 m (49 ft) tall in only 3 years. Its trunk is about 0.1–0.5 m (3.9–19.7 in) in diameter. It has yellow flowers.
###Biology

Mimosa scabrella (Bracatinga) is a tree in the subfamily Mimosoideae of the family Fabaceae. It is a cross-pollinating, mostly tetraploid plant with 52 chromosomes.

Mimosa scabrella is native to the southern region of Brazil. There it grows naturally in associations called “Bracatingais”. The Cerrado zone is a centre of biodiversity of Mimosa, where about one quarter of all Mimosa species are found. However M. scabrella evolved to grow in colder humid weather south from this region, in a sub-type of Atlantic Forest, called "mixed ombrophilous forest" (also known as Araucaria moist forests).

It is one of the fastest growing trees in the world. Within 14 months Mimosa scabrella grows up to 5 m (16 ft), in 2 years it reaches 8–9 m (26–30 ft), and in 3 years it can grow to a height of 15 m (49 ft).
####Characterization

The plant is characterized by quick growth, with a lean trunk of around 10–50 cm (3.9–19.7 in) in diameter. The leaves are bi-pinnate. Each leaf has several pinna, which again have 15–31 pairs of pinnules. The upper side of the leaves is yellow-green coloured with a paler underneath.

The flowers with ovary, narrow and slender pistils are ordered in clusters of 1–3 at the leaf bases. They are colored in a whitish to yellow color. Split-open pods are flattened, 2–4 mm (0.079–0.157 in) wide and 5–9 mm (0.20–0.35 in) long. They are covered with tiny warts and separated into 2–4 segments. Each segment is 4-angeled and 1-seeded. The seeds are little, brown, beanlike and about 3–6 mm (0.12–0.24 in) long. The dominant reproductive system is an allogamy (cross-pollinating) system. The reproductive age of M. scabrella is reached after around 3 years.
###Uses

Because of the abundant flowering and presence of honeydew caused by Cochineal infestation in some altitudes the tree has an important place in honey production, especially in Brazil. Its wood is suitable for firewood and can also be used as lumber. Before the advent of the diesel locomotive, M. scabrella wood was grown to fuel railroads in parts of Brazil. The long fibres are used for paper production.

In agroforestry M. scabrella shades coffee plants. It is also used in intercropping systems in association with maize and beans. Because M. scabrella is a legume tree it doesn't need fertilization and with the decomposition of the leaves, big amounts of nitrogen become available for other plants. Because M. scabrella has beautiful “feather” leaves, it is often used as an ornamental tree or live fence.

Because of its fast growth its often used for reforestation management.
###Alkaloids

Mimosa scabrella contains the alkaloids tryptamine, N-methyltryptamine, N,N-dimethyltryptamine and N-methyltetrahydrocarboline in its bark.
###Cultivation

Mimosa scabrella can be grown at altitudes of 200–2,400 m (660–7,870 ft) with an annual mean temperature between 12 and 23 °C (54 and 73 °F). The annual precipitation should reach from 600 to 3,500 mm (24 to 138 in). The soil should be well drained. Acid soils with pH as low as 4.8 and soils with high aluminum content are tolerated. Waterlogged, compacted or severely degraded soils are not suitable for M. scabrella. Dry periods of up to four months can be tolerated. Mimosa scabrella is susceptible to strong winds.

Although M. scabrella is native to Brazil, it is cultivated in many South American, some African and South European countries. In its native range, some 28 species of insects are reported to attack M. scabrella. 
'''),
    SpeciesDetail('Myroxylon balsamum', '''
Myroxylon balsamum, Santos mahogany, is a species of tree in the family Fabaceae. It is native to tropical forests from Southern Mexico through the Amazon regions of Peru and Brazil at elevations of 200–690 metres (660–2,260 ft). Plants are found growing in well drained soil in evergreen humid forest. 
''', "https://en.wikipedia.org/wiki/Myroxylon_balsamum", '''
Myroxylon balsamum, Santos mahogany, is a species of tree in the family Fabaceae. It is native to tropical forests from Southern Mexico through the Amazon regions of Peru and Brazil at elevations of 200–690 metres (660–2,260 ft). Plants are found growing in well drained soil in evergreen humid forest.
###Varieties

According to the Germplasm Resources Information Network, there are two varieties:

    Myroxylon balsamum var. balsamum – Tolu balsam tree
    Myroxylon balsamum var. pereirae (Royle) Harms – Peru balsam tree

###Description 

The tree is large slow growing, reaching 45 metres (148 ft) in height. Crown is round with dense foliage and the bark is yellowish with a pungent odor. Leaves alternate, petiolate, 8–20 centimetres (3–8 in) including the petioles, the petioles 1–4 centimetres (1⁄2–1+1⁄2 in) long, and the rachis 5–15 centimetres (2–6 in) long. The rachis and petioles are pubescent and terete. Leaflets are acute to acuminate at the apex, obtuse at the base, glabrous, with an entire margin and glandular oil dots.

Plants bloom 5 years from seeds during the months of February to June. Flowers are white, pubescent and has around 10 stamens. The fruit is a flat winged pod, narrowly obovate 8 centimetres (3+1⁄4 in) long 1–2 centimetres (3⁄8–3⁄4 in) wide, yellow to brown when dried and drop around November to January.

###Uses

Balsam of Tolu and Balsam of Peru are the products of the species' resin. They are extracted from different varieties in different ways. They are marketed among a series of intermediaries and exporters, their destinations being Germany, the United States of America, England and Spain, where they are used in the manufacture of cosmetics and medicines (for diseases of the skin, bronchi, lungs and airways, and in the treatment of burns and wounds). The tree has been planted for Balsam production in West Africa, India, and Sri Lanka.

The wood is reddish and has interlocked grain, which gives it a strong ribbon-like pattern, and logs produce a large amount of knot-free lumber. The wood has a Janka hardness of 2,200 pounds-force (9,800 N) and is somewhat resistant to fungal decay. Myroxylon balsamum wood is used for flooring, furniture, interior trim, and heavy construction.

M. balsamum is often used as a shade tree in coffee plantations. 
'''),
    SpeciesDetail('Ocotea porosa', '''
Ocotea porosa, commonly called imbuia or Brazilian walnut, is a species of plant in the Lauraceae family. Its wood is very hard, and it is a major commercial timber species in Brazil. 
''', "https://en.wikipedia.org/wiki/Ocotea_porosa", '''
Ocotea porosa, commonly called imbuia or Brazilian walnut, is a species of plant in the Lauraceae family. Its wood is very hard, and it is a major commercial timber species in Brazil.
###Taxonomy and naming

It is often placed in the related genus, Phoebe. It is commonly called imbuia, and is also known as Brazilian walnut, because its wood resembles that of some walnuts (to which it is not related).

Portuguese common names (with variant spellings) include embuia, embúia, embuya, imbuia, imbúia, imbuya, canela-imbuia.
###Habitat

The tree grows naturally in the subtropical montane Araucaria forests of southern Brazil, mostly in the states of Paraná and Santa Catarina (where it is the official state tree since 1973), and in smaller numbers in São Paulo and Rio Grande do Sul. The species may also occur in adjacent Argentina and/or Paraguay.

In its native habitat it is a threatened species.
###Description

The trees typically reach 40 m (130 ft) in height and 1.8 m (5 ft 11 in) in trunk diameter.

The wood is very hard, measuring 3,684 lbf (16,390 N) on the Janka scale. The wood is also fragrant with hints of nutmeg and cinnamon (also a member of the Lauraceae).
###Uses

The tree is a major commercial timber species in Brazil, used for high-end furniture, mostly as decorative veneers, and as flooring.

The tree is also a popular horticultural tree in subtropical regions of the world. 
'''),
    SpeciesDetail('Peltogyne sp', '''
Peltogyne, commonly known as purpleheart, violet wood, amaranth and other local names (often referencing the colour of the wood) is a genus of 23 species of flowering plants in the family Fabaceae; native to tropical rainforests of Central and South America; from Guerrero, Mexico, through Central America, and as far as south-eastern Brazil.
They are medium-sized to large trees growing to 30–50 m (100–160 ft) tall, with trunk diameters of up to 1.5 m (5 ft). The leaves are alternate, divided into a symmetrical pair of large leaflets 5–10 cm (2–4 in) long and 2–4 cm (1–2 in) broad. The flowers are small, with five white petals, produced in panicles. The fruit is a pod containing a single seed. The timber is desirable, but difficult to work.
''', "https://en.wikipedia.org/wiki/Peltogyne", '''
Peltogyne, commonly known as purpleheart, violet wood, amaranth and other local names (often referencing the colour of the wood) is a genus of 23 species of flowering plants in the family Fabaceae; native to tropical rainforests of Central and South America; from Guerrero, Mexico, through Central America, and as far as south-eastern Brazil.

They are medium-sized to large trees growing to 30–50 m (100–160 ft) tall, with trunk diameters of up to 1.5 m (5 ft). The leaves are alternate, divided into a symmetrical pair of large leaflets 5–10 cm (2–4 in) long and 2–4 cm (1–2 in) broad. The flowers are small, with five white petals, produced in panicles. The fruit is a pod containing a single seed. The timber is desirable, but difficult to work.
###Distribution

The species of the genus range from southeastern Brazil through northern South America, Panama, Costa Rica, and Trinidad, with the majority of species in the Amazon Basin. P. mexicana is a geographic outlier, native to the Mexican state of Guerrero. Overharvesting has caused several species to become endangered in areas where they were once abundant.
###Wood

The trees are prized for their beautiful heartwood which, when cut, quickly turns from a light brown to a rich purple color. Exposure to ultraviolet (UV) light darkens the wood to a brown color with a slight hue of the original purple. This effect can be minimized with a finish containing a UV inhibitor. The dry timber is very hard, stiff, and dense with a specific gravity of 0.86 (860 kg/m3 or 54 lb/cu ft). Purpleheart is correspondingly difficult to work with. It is very durable and water-resistant.
###Uses and hazards

Purpleheart is prized for use in fine inlay work especially on musical instruments, guitar fret boards (although rarely), woodturning, cabinetry, flooring, and furniture.

Purpleheart presents a number of challenges in the woodshop. Its hard-to-detect interlocking grain makes hand-planing, chiseling and working with carving tools a challenge. However, woodturners can note that with sharp tools, it turns clean, and sands well.

Exposure to the dust generated by cutting and sanding purpleheart can cause skin, eye, and respiratory irritation and nausea, possibly because of the presence of dalbergione (neoflavonoid) compounds in the wood. This also makes purpleheart wood unsuitable to most people for use in jewelry. Purpleheart is also a fairly expensive wood, which is why it is usually used in smaller-scale projects. 
'''),
    SpeciesDetail('Pinus sp', '''
A pine is any conifer tree or shrub in the genus Pinus (/ˈpaɪnuːs/) of the family Pinaceae. Pinus is the sole genus in the subfamily Pinoideae.
World Flora Online, created by the Royal Botanic Gardens, Kew, and Missouri Botanical Garden, accepts 187 species names of pines as current, together with more synonyms. The American Conifer Society (ACS) and the Royal Horticultural Society accept 121 species.
Pines are commonly found in the Northern Hemisphere.
Pine may also refer to the lumber derived from pine trees; it is one of the more extensively used types of lumber.
The pine family is the largest conifer family, and there are currently 818 named cultivars (or trinomials) recognized by the ACS. It is also a well-known type of Christmas tree. 
''', "https://en.wikipedia.org/wiki/Pine", '''
A pine is any conifer tree or shrub in the genus Pinus (/ˈpaɪnuːs/) of the family Pinaceae. Pinus is the sole genus in the subfamily Pinoideae.

World Flora Online, created by the Royal Botanic Gardens, Kew, and Missouri Botanical Garden, accepts 187 species names of pines as current, together with more synonyms. The American Conifer Society (ACS) and the Royal Horticultural Society accept 121 species.

Pines are commonly found in the Northern Hemisphere.

Pine may also refer to the lumber derived from pine trees; it is one of the more extensively used types of lumber.

The pine family is the largest conifer family, and there are currently 818 named cultivars (or trinomials) recognized by the ACS. It is also a well-known type of Christmas tree.
###Description

Pine trees are evergreen, coniferous resinous trees (or, rarely, shrubs) growing 3–80 metres (10–260 feet) tall, with the majority of species reaching 15–45 m (50–150 ft) tall. The smallest are Siberian dwarf pine and Potosi pinyon, and the tallest is an 81.8 m (268 ft) tall ponderosa pine located in southern Oregon's Rogue River-Siskiyou National Forest.

Pines are long lived and typically reach ages of 100–1,000 years, some even more. The longest-lived is the Great Basin bristlecone pine (P. longaeva). One individual of this species, dubbed "Methuselah", is one of the world's oldest living organisms at around 4,800 years old. This tree can be found in the White Mountains of California. An older tree, now cut down, was dated at 4,900 years old. It was discovered in a grove beneath Wheeler Peak and it is now known as "Prometheus" after the Greek immortal.

The spiral growth of branches, needles, and cones scales may be arranged in Fibonacci number ratios. The new spring shoots are sometimes called "candles"; they are covered in brown or whitish bud scales and point upward at first, then later turn green and spread outward. These "candles" offer foresters a means to evaluate fertility of the soil and vigour of the trees.
####Bark

The bark of most pines is thick and scaly, but some species have thin, flaky bark. The branches are produced in regular "pseudo whorls", actually a very tight spiral but appearing like a ring of branches arising from the same point. Many pines are uninodal, producing just one such whorl of branches each year, from buds at the tip of the year's new shoot, but others are multinodal, producing two or more whorls of branches per year.
####Foliage

Pines have four types of leaf:

    Seed leaves (cotyledons) on seedlings are borne in a whorl of 4–24.
    Juvenile leaves, which follow immediately on seedlings and young plants, are 2–6 centimetres (3⁄4–2+1⁄4 inches) long, single, green or often blue-green, and arranged spirally on the shoot. These are produced for six months to five years, rarely longer.
    Scale leaves, similar to bud scales, are small, brown and not photosynthetic, and arranged spirally like the juvenile leaves.
    Needles, the adult leaves, are green (photosynthetic) and bundled in clusters called fascicles. The needles can number from one to seven per fascicle, but generally number from two to five. Each fascicle is produced from a small bud on a dwarf shoot in the axil of a scale leaf. These bud scales often remain on the fascicle as a basal sheath. The needles persist for 1.5–40 years, depending on species. If a shoot's growing tip is damaged (e.g. eaten by an animal), the needle fascicles just below the damage will generate a stem-producing bud, which can then replace the lost growth tip.

####Cones

Pines are monoecious, having the male and female cones on the same tree.: 205  The male cones are small, typically 1–5 cm long, and only present for a short period (usually in spring, though autumn in a few pines), falling as soon as they have shed their pollen. The female cones take 1.5–3 years (depending on species) to mature after pollination, with actual fertilization delayed one year. At maturity the female cones are 3–60 cm long. Each cone has numerous spirally arranged scales, with two seeds on each fertile scale; the scales at the base and tip of the cone are small and sterile, without seeds.

The seeds are mostly small and winged, and are anemophilous (wind-dispersed), but some are larger and have only a vestigial wing, and are bird-dispersed. Female cones are woody and sometimes armed to protect developing seeds from foragers. At maturity, the cones usually open to release the seeds. In some of the bird-dispersed species, for example whitebark pine, the seeds are only released by the bird breaking the cones open. In others, the seeds are stored in closed cones for many years until an environmental cue triggers the cones to open, releasing the seeds. This is called serotiny. The most common form of serotiny is pyriscence, in which a resin binds the cones shut until melted by a forest fire, for example in P. rigida.
###Taxonomy

Pines are gymnosperms. The genus is divided into two subgenera based on the number of fibrovascular bundles in the needle. The subgenera can be distinguished by cone, seed, and leaf characters:

    Pinus subg. Pinus, the yellow, or hard pine group, generally with harder wood and two or three needles per fascicle. The subgenus is also named diploxylon, on account of its two fibrovascular bundles.
    Pinus subg. Strobus, the white, or soft pine group. Its members usually have softer wood and five needles per fascicle. The subgenus is also named haploxylon, on account of its one fibrovascular bundle.

Phylogenetic evidence indicates that both subgenera have a very ancient divergence from one another. Each subgenus is further divided into sections and subsections.

Many of the smaller groups of Pinus are composed of closely related species with recent divergence and history of hybridization. This results in low morphological and genetic differences. This, coupled with low sampling and underdeveloped genetic techniques, has made taxonomy difficult to determine. Recent research using large genetic datasets has clarified these relationships into the groupings we recognize today.
####Etymology

The modern English name "pine" derives from Latin pinus, which some have traced to the Indo-European base *pīt- ‘resin’ (source of English pituitary). Before the 19th century, pines were often referred to as firs (from Old Norse fura, by way of Middle English firre). In some European languages, Germanic cognates of the Old Norse name are still in use for pines — in Danish fyr, in Norwegian fura/fure/furu, Swedish fura/furu, Dutch vuren, and German Föhre — but in modern English, fir is now restricted to fir (Abies) and Douglas-fir (Pseudotsuga).
####Phylogeny

Pinus is the largest genus of the Pinaceae, the pine family, which first appeared in the Jurassic period. Based on recent Transcriptome analysis, Pinus is most closely related to the genus Cathaya, which in turn is closely related to spruces. These genera, with firs and larches, form the pinoid clade of the Pinaceae. Pines first appeared during the Early Cretaceous, with the oldest verified fossil of the genus being Pinus yorkshirensis from the Hauterivian-Barremian boundary (~130-125 million years ago) from the Speeton Clay, England.

The evolutionary history of the genus Pinus has been complicated by hybridization. Pines are prone to inter-specific breeding. Wind pollination, long life spans, overlapping generations, large population size, and weak reproductive isolation make breeding across species more likely. As the pines have diversified, gene transfer between different species has created a complex history of genetic relatedness. 

####Distribution and habitat

Pines are native to the Northern Hemisphere, and to a few parts from the tropics to temperate regions in the Southern Hemisphere. Most regions of the Northern Hemisphere host some native species of pines. One species (Sumatran pine) crosses the equator in Sumatra to 2°S. In North America, various species occur in regions at latitudes from as far north as 66°N to as far south as 12°N.

Pines may be found in a very large variety of environments, ranging from semi-arid desert to rainforests, from sea level up to 5,200 m (17,100 ft), from the coldest to the hottest environments on Earth. They often occur in mountainous areas with favorable soils and at least some water.

Various species have been introduced to temperate and subtropical regions of both hemispheres, where they are grown as timber or cultivated as ornamental plants in parks and gardens. A number of such introduced species have become naturalized, and some species are considered invasive in some areas and threaten native ecosystems.
###Ecology
Pine beauty moth (Panolis flammea) on pine needles

Pines grow well in acid soils, some also on calcareous soils; most require good soil drainage, preferring sandy soils, but a few (e.g. lodgepole pine) can tolerate poorly drained wet soils. A few are able to sprout after forest fires (e.g. Canary Island pine). Some species of pines (e.g. bishop pine) need fire to regenerate, and their populations slowly decline under fire suppression regimens.

Pine trees are beneficial to the environment since they can remove carbon dioxide from the atmosphere. Although several studies have indicated that after the establishment of pine plantations in grasslands, there is an alteration of carbon pools including a decrease of the soil organic carbon pool.

Several species are adapted to extreme conditions imposed by elevation and latitude (e.g. Siberian dwarf pine, mountain pine, whitebark pine, and the bristlecone pines). The pinyon pines and a number of others, notably Turkish pine and gray pine, are particularly well adapted to growth in hot, dry semidesert climates.

Pine pollen may play an important role in the functioning of detrital food webs. Nutrients from pollen aid detritivores in development, growth, and maturation, and may enable fungi to decompose nutritionally scarce litter. Pine pollen is also involved in moving plant matter between terrestrial and aquatic ecosystems.
####Wildlife

Pine needles serve as food for various Lepidoptera (butterfly and moth) species. Several species of pine are attacked by nematodes, causing pine wilt disease, which can kill some quickly. Some of these Lepidoptera species, many of them moths, specialize in feeding on only one or sometimes several species of pine. Beside that many species of birds and mammals shelter in pine habitat or feed on pine nuts.

The seeds are commonly eaten by birds, such as grouse, crossbills, jays, nuthatches, siskins, and woodpeckers, and by squirrels. Some birds, notably the spotted nutcracker, Clark's nutcracker, and pinyon jay, are of importance in distributing pine seeds to new areas. Pine needles are sometimes eaten by the Symphytan species pine sawfly, and goats.
###Uses
####Lumber and construction

Pines are among the most commercially important tree species valued for their timber and wood pulp throughout the world. In temperate and tropical regions, they are fast-growing softwoods that grow in relatively dense stands, their acidic decaying needles inhibiting the sprouting of competing hardwoods. Commercial pines are grown in plantations for timber that is denser and therefore more durable than spruce (Picea). Pine wood is widely used in high-value carpentry items such as furniture, window frames, panelling, floors, and roofing, and the resin of some species is an important source of turpentine.

Because pine wood has no insect- or decay-resistant qualities after logging, in its untreated state it is generally recommended for indoor construction purposes only (indoor drywall framing, for example). For outside use, pine needs to be treated with copper azole, chromated copper arsenate or other suitable chemical preservative.
####Ornamental uses

Many pine species make attractive ornamental plantings for parks and larger gardens with a variety of dwarf cultivars being suitable for smaller spaces. Pines are also commercially grown and harvested for Christmas trees. Pine cones, the largest and most durable of all conifer cones, are craft favorites. Pine boughs, appreciated especially in wintertime for their pleasant smell and greenery, are popularly cut for decorations. Pine needles are also used for making decorative articles such as baskets, trays, pots, etc., and during the U.S. Civil War, the needles of the longleaf pine "Georgia pine" were widely employed in this. This originally Native American skill is now being replicated across the world. Pine needle handicrafts are made in the US, Canada, Mexico, Nicaragua, and India. Pine needles are also versatile and have been used by Latvian designer Tamara Orjola to create different biodegradable products including paper, furniture, textiles and dye.
####Farming

When grown for sawing timber, pine plantations can be harvested after 25 years, with some stands being allowed to grow up to 50 (as the wood value increases more quickly as the trees age). Imperfect trees (such as those with bent trunks or forks, smaller trees, or diseased trees) are removed in a "thinning" operation every 5–10 years. Thinning allows the best trees to grow much faster, because it prevents weaker trees from competing for sunlight, water, and nutrients. Young trees removed during thinning are used for pulpwood or are left in the forest, while most older ones are good enough for saw timber.

A 30-year-old commercial pine tree grown in good conditions in Arkansas will be about 0.3 m (1 ft) in diameter and about 20 m (66 ft) high. After 50 years, the same tree will be about 0.5 m (1+1⁄2 ft) in diameter and 25 m (82 ft) high, and its wood will be worth about seven times as much as the 30-year-old tree. This however depends on the region, species and silvicultural techniques. In New Zealand, a plantation's maximum value is reached after around 28 years with height being as high as 30 m (98 ft) and diameter 0.5 m (1+1⁄2 ft), with maximum wood production after around 35 years (again depending on factors such as site, stocking and genetics). Trees are normally planted 3–4 m apart, or about 1,000 per hectare (100,000 per square kilometre).
####Food and nutrients

The seeds (pine nuts) are generally edible; the young male cones can be cooked and eaten, as can the bark of young twigs. Some species have large pine nuts, which are harvested and sold for cooking and baking. They are an essential ingredient of pesto alla genovese.

The soft, moist, white inner bark (cambium) beneath the woody outer bark is edible and very high in vitamins A and C. It can be eaten raw in slices as a snack or dried and ground up into a powder for use as an ersatz flour or thickener in stews, soups, and other foods, such as bark bread. Adirondack Indians got their name from the Mohawk Indian word atirú:taks, meaning "tree eaters".

A tea is made by steeping young, green pine needles in boiling water (known as tallstrunt in Sweden). In eastern Asia, pine and other conifers are accepted among consumers as a beverage product, and used in teas, as well as wine. In Greece, the wine retsina is flavoured with Aleppo pine resin.

Pine needles from Pinus densiflora were found to contain 30.54 milligram/gram of proanthocyanidins when extracted with hot water. Comparative to ethanol extraction resulting in 30.11 mg/g, simply extracting in hot water is preferable.

In traditional Chinese medicine, pine resin is used for burns, wounds and dermal complaints.
###Culture

Pines have been a frequently mentioned tree throughout history, including in literature, paintings and other art, and in religious texts.
####Literature

Writers of various nationalities and ethnicities have written of pines. Among them, John Muir, Dora Sigerson Shorter, Eugene Field, Bai Juyi, Theodore Winthrop, and Rev. George Allan D.D.
####Art

Pines are often featured in art, whether painting and fine art, drawing, photography, or folk art.
####Religious texts

Pine trees, as well as other conifers, are mentioned in some verses of the Bible, depending on the translation. In the Book of Nehemiah 8:15, the King James Version gives the following translation:

    "And that they should publish and proclaim in all their cities, and in Jerusalem, saying, Go forth unto the mount, and fetch olive branches, and pine branches [emphasis added], and myrtle branches, and palm branches, and branches of thick trees, to make booths, as it is written."

However, the term here in Hebrew (עץ שמן) means "oil tree" and it is not clear what kind of tree is meant. Pines are also mentioned in some translations of Isaiah 60:13, such as the King James:

    "The glory of Lebanon shall come unto thee, the fir tree, the pine tree, and the box together, to beautify the place of my sanctuary; and I will make the place of my feet glorious."

Again, it is not clear what tree is meant (תדהר in Hebrew), and other translations use "pine" for the word translated as "box" by the King James (תאשור in Hebrew).

Some botanical authorities believe that the Hebrew word "ברוש" (bərōsh), which is used many times in the Bible, designates P. halepensis, or in Hosea 14:8 which refers to fruit, Pinus pinea, the stone pine.  The word used in modern Hebrew for pine is "אֹ֖רֶן" (oren), which occurs only in Isaiah 44:14, but two manuscripts have "ארז" (cedar), a much more common word.
####Chinese culture

The pine is a motif in Chinese art and literature, which sometimes combines painting and poetry in the same work. Some of the main symbolic attributes of pines in Chinese art and literature are longevity and steadfastness: the pine retains its green needles through all the seasons. Sometimes the pine and cypress are paired. At other times the pine, plum, and bamboo are considered as the "Three Friends of Winter". Many Chinese art works and/or literature (some involving pines) have been done using paper, brush, and Chinese ink: interestingly enough, one of the main ingredients for Chinese ink has been pine soot. 
'''),
    SpeciesDetail('Podocarpus lambertii', '''
Podocarpus lambertii is a species of conifer in the family Podocarpaceae. It is found in Argentina and Brazil. 
''', "https://en.wikipedia.org/wiki/Podocarpus_lambertii", '''
'''),
    SpeciesDetail('Pouteria pachycarpa', '''
A timber species from South and Central America. Commonly used in Heavy carpentry, Cabinetwork (high class furniture), Interior panelling, Turned goods, Flooring, Interior joinery, Sliced veneer, Tool handles (resilient woods)
''', "https://www.lesserknowntimberspecies.com/species/goiabao", '''
'''),
    SpeciesDetail('Swietenia macrophylla', '''
Swietenia macrophylla, commonly known as mahogany, Honduran mahogany, Honduras mahogany, or big-leaf mahogany is a species of plant in the Meliaceae family. It is one of three species that yields genuine mahogany timber (Swietenia), the others being Swietenia mahagoni and Swietenia humilis. It is native to South America, Mexico and Central America, but naturalized in the Philippines, Singapore, Malaysia and Hawaii, and cultivated in plantations and wind-breaks elsewhere. 
''', "https://en.wikipedia.org/wiki/Swietenia_macrophylla", '''
Swietenia macrophylla, commonly known as mahogany, Honduran mahogany, Honduras mahogany, or big-leaf mahogany is a species of plant in the Meliaceae family. It is one of three species that yields genuine mahogany timber (Swietenia), the others being Swietenia mahagoni and Swietenia humilis. It is native to South America, Mexico and Central America, but naturalized in the Philippines, Singapore, Malaysia and Hawaii, and cultivated in plantations and wind-breaks elsewhere.
###Description
####Wood

Mahogany wood is strong and is usually a source for furniture, musical instruments, ships, doors, coffins, decors.
####Leaves

Mahogany is characterised by its large leaves (up to 45 cm long). The leaflets are even in number and are connected by a central midrib.
####Fruits

The fruits are called "sky fruits" because of its upwards growth towards the sky. The fruits of mahogany can be measure to 40 cm in length, in a light grey to brown capsule. Each fruit capsule could contain 71 winged seeds.
####Seeds

The seeds of mahogany can reach 7 to 12 cm long.
###Timber

Unlike mahogany sourced from its native locations, plantation mahogany grown in Asia is not restricted in trade. The mahogany timber grown in these Asian plantations is the major source of international trade in genuine mahogany today. The Asian countries which grow the majority of Swietenia macrophylla are India, Indonesia, Malaysia, Bangladesh, Fiji, Philippines, Singapore, and some others, with India and Fiji being the major world suppliers. The tree is also planted in Laos PDR.
###Medicinal use

It was scientifically studied for its various biological activities.  A detailed mechanism of action of apoptotic inducing effect on HCT116 human cancer cell line was elucidated.  Through solvent extraction and fractionation done on seeds of Swietenia macrophylla, the ethyl acetate fraction (SMEAF) was further examined for its neuroprotective activity and acute toxicity effects.  Various purified compounds derived from Swietenia macrophylla were further examined and was revealed to possesses potent PPARγ binding activity which might capable of stimulating glucose uptake in muscle cells. 

The ethyl acetate fraction from the seeds of Swietenia macrophylla (SMEAF) was studied for anti-inflammatory properties using lipopolysaccharide (LPS)-induced BV-2 microglia. SMEAF significantly attenuated the LPS-induced production of nitric oxide (NO), inducible nitric oxide synthase (iNOS), cyclooxygenase-2 (COX-2), tumour necrosis factor-α (TNF-α) and interleukin-6 (IL-6). SMEAF inhibited nuclear translocation of nuclear factor-kappa B (NF-κB) via the attenuation of IκBα phosphorylation. Moreover, SMEAF markedly suppressed phosphorylation of Akt, p38 Mitogen-activated protein kinase (MAPK) and Extracellular signal-regulated kinase 1/2 (ERK1/2) in LPS-induced BV-2 cells. Treatment with specific inhibitors for Akt, NF-κB, p38 and ERK1/2 resulted in the attenuation of iNOS and COX-2 protein expression. These findings indicated that SMEAF possesses anti-inflammatory activities in BV-2 cells by modulating LPS-induced pro-inflammatory mediator production via the inhibition of Akt-dependent NF-κB, p38 MAPK and ERK1/2 activation. These results further advocate the potential use of S. macrophylla as nutraceutical for the intervention of neurodegenerative and neuroinflammatory disorders.

There are also claims of its ability to improve blood circulation and skin condition, as well as anti-erectile dysfunction.

However, there are reports of liver injury or hepatotoxicity after consumption of Mahogany Seeds both in raw form and raw seeds grind and pack in capsule form. The severity of liver damage varies. There are also the report of single case kidney injury and polyarthralgia. In most cases, the liver function was recovered after stopping the consumption. The exact mechanism of these adverse events is currently unknown.

These cases that happened are the first reports of Swietenia Macrophylla seeds’ association with liver injury. This may also due to over dosage and consumption of contaminated raw seeds which are never been thoroughly investigated. Based on acute oral toxicity studies of Swietenia Macrophylla seeds, the consumption of Swietenia Macrophylla by humans is safe if the dose is less than 325 mg/kg body weight. The usual dose of Swietenia Macrophylla prescribed in Malaysian folk-lore medicine is one seed per day.
###Population genetics

Mesoamerican rainforest populations show higher structure than in the Amazon.
###Common names

The species is also known under other common names, including bastard mahogany, broad-leaved mahogany, Brazilian mahogany, large-leaved mahogany, genuine mahogany, tropical American mahogany, and sky fruit, among others.

    English - big leaf mahogany, large-leaved mahogany, Brazilian mahogany
    French - acajou à grandes feuilles, acajou du Honduras
    Spanish - caoba, mara, mogno
    Malayalam - mahagony
    Tamil - Thenkani (தேன்கனி)
    Telugu - mahagani, peddakulamaghani
    Sinhala - mahogani (මහෝගනි)
'''),
    SpeciesDetail('Tabebuia sp', '''
Tabebuia is a genus of flowering plants in the family Bignoniaceae. Tabebuia consists almost entirely of trees, but a few are often large shrubs. A few species produce timber, but the genus is mostly known for those that are cultivated as flowering trees. 
''', "https://en.wikipedia.org/wiki/Tabebuia", '''
Tabebuia is a genus of flowering plants in the family Bignoniaceae. Tabebuia consists almost entirely of trees, but a few are often large shrubs. A few species produce timber, but the genus is mostly known for those that are cultivated as flowering trees.
###Etymology

The genus name is derived from the Tupi words for "ant" and "wood", referring to the fact that many Tabebuia species have twigs with soft pith which forms hollows within which ants live, defending the trees from other herbivores. The ants are attracted to the plants by special extra-floral nectar glands on at the apex of the petioles. The common name "roble" is sometimes found in English. Tabebuias have been called "trumpet trees", but this name is usually applied to other trees and has become a source of confusion and misidentification.
###Distribution

Tabebuia is native to the American tropics and subtropics from Mexico and the Caribbean to Argentina. Most of the species known are from the islands of Cuba and Hispaniola. It is commonly cultivated and often naturalized or adventive beyond its natural range. It easily escapes cultivation because of its numerous, air-borne seeds.
###Taxonomy

In 1992, a revision of Tabebuia described 99 species and one hybrid. Phylogenetic studies of DNA sequences later showed that Tabebuia, as then circumscribed, was polyphyletic. In 2007, it was divided into three separate genera. Primavera (Roseodendron donnell-smithii) and a related species with no unique common name (Roseodendron chryseum) were transferred to Roseodendron. Those species known as ipê and pau d'arco (in Portuguese) or poui were transferred to Handroanthus. Sixty-seven species remained in Tabebuia. The former genus and polyphyletic group of 99 species described by Gentry in 1992 is now usually referred to as "Tabebuia sensu lato".
####Species

All of the species in the first two columns below were recognized and described by Gentry in 1992. Listed in the third column are species names that have been used recently, but were not accepted by Gentry. The currently accepted synonym for each is in parentheses.

Some recently used names in Tabebuia that were not recognized by Gentry are not listed in the third column below because they apply to species that are now in Handroanthus. Tabebuia spectabilis is an obsolete name for Handroanthus chrysanthus subsp. meridionalis. Tabebuia ecuadorensis is now synonymized under Handroanthus billbergii. Tabebuia heteropoda is now synonymized under Handroanthus ochraceus.

####Taxonomic history

The name Tabebuia entered the botanical literature in 1803, when António Bernardino Gomes used it as a common name for Tabebuia uliginosa, now a synonym for Tabebuia cassinoides, which he described as a species of Bignonia. Tabebuia is an abbreviation of "tacyba bebuya", a Tupi name meaning "ant wood". Among the Indigenous peoples in Brazil, similar names exist for various species of Tabebuia.

Tabebuia was first used as a generic name by Augustin Pyramus de Candolle in 1838. The type species for the genus is Tabebuia uliginosa, which is now a synonym for Tabebuia cassinoides. Confusion soon ensued over the meaning of Tabebuia and what to include within it. Most of the misunderstanding was cleared up by Nathaniel Lord Britton in 1915. Britton revived the concept of Tabebuia that had been originated in 1876 by Bentham and Hooker, consisting of species with either simple or palmately compound leaves. Similar plants with pinnately compound leaves were placed in Tecoma. This is the concept of Tabebuia that was usually followed until 2007.

The genus Roseodendron was established by Faustino Miranda González in 1965 for the two species now known as Roseodendron donnell-smithii and Roseodendron chryseum. These species had been placed in Cybistax by Russell J. Seibert in 1940, but were returned to Tabebuia by Alwyn H. Gentry in 1992.

Handroanthus was established by Joáo Rodrigues de Mattos in 1970. Gentry did not agree with the segregation of Handroanthus from Tabebuia and warned against "succumbing to further paroxysms of unwarranted splitting". In 1992, Gentry published a revision of Tabebuia in Flora Neotropica, in which he described 99 species and one hybrid, including those species placed by some authors in Roseodendron or Handroanthus. Gentry divided Tabebuia into ten "species groups", some of them intentionally artificial. Tabebuia, as currently circumscribed, consists of groups 2, 6, 7, 8, 9, and 10. Group 1 is now the genus Roseodendron. Groups 3, 4, and 5 compose the genus Handroanthus.

In 2007, a molecular phylogenetic study found Handroanthus to be closer to a certain group of four genera than to Tabebuia. This group consists of Spirotecoma, Parmentiera, Crescentia, and Amphitecna. A phylogenetic tree can be seen at Bignoniaceae. Handroanthus was duly resurrected and 30 species were assigned to it, with species boundaries the same as those of Gentry (1992).

Roseodendron was resolved as sister to a clade consisting of Handroanthus and four other genera. This result had only weak statistical support, but Roseodendron clearly did not group with the remainder of Tabebuia. Consequently, Roseodendron was resurrected in its original form. The remaining 67 species of Tabebuia formed a strongly supported clade that is sister to Ekmanianthe, a genus of two species from Cuba and Hispaniola. Tabebuia had been traditionally placed in the tribe Tecomeae, but that tribe is now defined much more narrowly than it had been, and it now excludes Tabebuia. Tabebuia is now one of 12 to 14 genera belonging to a group that is informally called the Tabebuia alliance. This group has not been placed at any particular taxonomic rank.

Cladistic analysis of DNA data has strongly supported Tabebuia by Bayesian inference and maximum parsimony. Such studies have so far revealed almost nothing about relationships within the genus, placing nearly all of the sampled species in a large polytomy.
###Description

The description below is excerpted from Grose and Olmstead (2007).

    Trees or shrubs. Evergreen or dry season deciduous.
    Wood lacking lapachol; not especially dense or hard. Heartwood light brown to reddish brown, not distinct from sapwood.
    Leaves sometimes simple; usually palmately 3 to 7(9)-foliate; with stalked or sessile lepidote scales.
    Inflorescences usually few-flowered panicles, dichotomously branching, without a well-developed, central rachis.
    Calyx coriaceous, spathaceous; irregularly 2 to 3-labiate, rarely 5-dentate.
    Corolla yellow in two species (T. aurea and T. nodosa); otherwise white to pink, rarely red, often with a yellow throat.
    Stamens didynamous; staminode small.
    Ovary linear, bilocular.
    Ovules in two or three series in each locule.
    Fruit a dehiscent capsule, usually linear, sometimes ribbed, glabrous except for lepidote scales.
    Seeds thin, with two wings; wings hyaline, membranaceous, and sharply demarcated from the seed body.

Tabebuia is distinguished from Handroanthus by wood that is not especially hard or heavy, and not abruptly divided into heartwood and sapwood. Lapachol is absent. Scales are present, but no hair. The calyx is usually spathaceous in Tabebuia, but never so in Handroanthus. Only two species of Tabebuia are yellow-flowered, but most species of Handroanthus are.

Unlike Roseodendron, the calyx of Tabebuia is always distinctly harder and thicker than the corolla. Tabebuia always has a dichotomously branched inflorescence; never a central rachis as in Roseodendron. Some species of Tabebuia have ribbed fruit, but not as conspicuously so as the two species of Roseodendron.
###Uses

The wood of Tabebuia is light to medium in weight. Tabebuia rosea (including T. pentaphylla) is an important timber tree of tropical America. Tabebuia heterophylla and Tabebuia angustata are the most important timber trees of some of the Caribbean islands. Their wood is of medium weight and is exceptionally durable in contact with salt water.

The swamp species of Tabebuia have wood that is unusually light in weight. The most prominent example of these is Tabebuia cassinoides. Its roots produce a soft and spongy wood that is used for floats, razor strops, and the inner soles of shoes.

In spite of its use for lumber, Tabebuia is best known as an ornamental flowering tree. Tabebuia aurea, Tabebuia rosea, Tabebuia pallida, Tabebuia berteroi, and Tabebuia heterophylla are cultivated throughout the tropics for their showy flowers. Tabebuia dubia, Tabebuia haemantha, Tabebuia obtusifolia, Tabebuia nodosa, and Tabebuia roseo-alba are also known in cultivation and are sometimes locally abundant.

Some species of Tabebuia have been grown as honey plants by beekeepers. 
'''),
    SpeciesDetail('Virola surinamensis', '''
Virola surinamensis, known commonly as baboonwood, ucuuba, ucuhuba and chalviande, is a species of flowering plant in the family Myristicaceae. It is found in Brazil, Costa Rica, Ecuador, French Guiana, Guyana, Panama, Peru, Suriname, and Venezuela. It has also been naturalized in the Caribbean. Its natural habitats are subtropical or tropical moist lowland forests, subtropical or tropical swamps, and heavily degraded former forest. Although the species is listed as threatened due to habitat loss by the IUCN, it is a common tree species found throughout Central and South America.
''', "https://en.wikipedia.org/wiki/Virola_surinamensis", '''
Virola surinamensis, known commonly as baboonwood, ucuuba, ucuhuba and chalviande, is a species of flowering plant in the family Myristicaceae. It is found in Brazil, Costa Rica, Ecuador, French Guiana, Guyana, Panama, Peru, Suriname, and Venezuela. It has also been naturalized in the Caribbean. Its natural habitats are subtropical or tropical moist lowland forests, subtropical or tropical swamps, and heavily degraded former forest. Although the species is listed as threatened due to habitat loss by the IUCN, it is a common tree species found throughout Central and South America.

Virola surinamensis grows 25–40 m (82–131 ft) tall. The leaves are 10–22 cm (3.9–8.7 in) long and 2–5 cm (0.79–1.97 in) wide. The fruits are ellipsoidal to subglobular, measuring about 13–21 mm (0.51–0.83 in) long and 11–18 mm (0.43–0.71 in) in diameter.
###Uses

The tree is harvested for its wood. It is also a source of traditional medicinal remedies for intestinal worms. The Amazon Indians Waiãpi living in the West of Amapá State of Brazil, treat malaria with an inhalation of vapor obtained from leaves of Viola surinamensis.
####Ucuhuba seed oil

Ucuhuba seed oil is the oil extracted from the seed. It contains 13% lauric acid, 69% myristic acid, 7% palmitic acid, and traces of oleic acid and linoleic acid. Myristic and lauric acids comprised 91.3 mole % of the total fatty acids. Additional saturated fatty acids such as decanoic acid and stearic acid are minor components. 
'''),
    SpeciesDetail('Vochysia sp', '''
Vochysia is a genus of plant in the family Vochysiaceae. It contains the following species, among many others:
Vochysia aurifera, Standl. & L.O. Williams
Vochysia haenkeana
Vochysia wilsonii Marc.-Berti, J.M. Vélez. & Aymard, 2023
''', "https://en.wikipedia.org/wiki/Vochysia", '''
'''),
]