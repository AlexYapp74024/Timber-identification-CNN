class SpeciesDetail():
    def __init__(self, name, desc, link) -> None:
        self.name = name
        self.desc = desc
        self.link = link

labels = [
    SpeciesDetail('Araucaria angustifolia', '''
Araucaria angustifolia, the Paraná pine, Brazilian pine or candelabra tree (pinheiro-do-paraná, araucária or pinheiro brasileiro), is a critically endangered species in the conifer genus Araucaria. Although the common names in various languages refer to the species as a "pine", it does not belong in the genus Pinus. 
''', "https://en.wikipedia.org/wiki/Araucaria_angustifolia"),
    SpeciesDetail('Aspidosperma polyneuron', '''
Aspidosperma polyneuron is a timber tree native to Brazil, Colombia, Peru, Argentina, and Paraguay. It is common in Atlantic Forest vegetation. In addition, it is useful for beekeeping. 
''', "https://en.wikipedia.org/wiki/Aspidosperma_polyneuron"),
    SpeciesDetail('Bagassa guianensis', '''
Bagassa guianensis is a tree in the plant family Moraceae which is native to the Guianas and Brazil. It is valued as a timber tree and as a food tree for wildlife. The juvenile leaves are distinctly different in appearance from the mature leaves, and were once thought to belong to different species. 
''', "https://en.wikipedia.org/wiki/Bagassa"),
    SpeciesDetail('Balfourodendron riedelianum', '''
Balfourodendron riedelianum, known as marfim in Portuguese, is a species of flowering tree in the rue family, Rutaceae. It is native to Argentina, Brazil, and Paraguay. 
''', "https://en.wikipedia.org/wiki/Balfourodendron_riedelianum"),
    SpeciesDetail('Bertholethia excelsa', '''
The Brazil nut (Bertholletia excelsa) is a South American tree in the family Lecythidaceae, and it is also the name of the tree's commercially harvested edible seeds. It is one of the largest and longest-lived trees in the Amazon rainforest. The fruit and its nutshell - containing the edible Brazil nut - are relatively large, possibly weighing as much as 2 kg (4.4 lb) in total weight. As food, Brazil nuts are notable for diverse content of micronutrients, especially a high amount of selenium. The wood of the Brazil nut tree is prized for its quality in carpentry, flooring, and heavy construction. 
''', "https://en.wikipedia.org/wiki/Brazil_nut"),
    SpeciesDetail('Bowdichia sp', '''
Bowdichia is a genus of flowering plants in the legume family, Fabaceae. It belongs to the subfamily Faboideae. The genus includes two species native to tropical South America and Costa Rica. 
''', "https://en.wikipedia.org/wiki/Bowdichia"),
    SpeciesDetail('Brosimum paraense', '''
Brosimum is a genus of plants in the family Moraceae, native to tropical regions of the Americas.
The breadnut (B. alicastrum) was used by the Maya civilization for its edible nut. The dense vividly colored scarlet wood of B. paraense is used for decorative woodworking. B. guianense, or snakewood, has a mottled snake-skin pattern, and is among the densest woods, with a very high stiffness; it was the wood of choice for making of bows for musical instruments of the violin family until the late 18th century, when it was replaced by the more easily worked brazilwood (Paubrasilia echinata). Plants of this genus are otherwise used for timber, building materials, and in a cultural context. 
''', "https://en.wikipedia.org/wiki/Brosimum"),
    SpeciesDetail('Carapa guianensis', '''
Carapa guianensis is a species of tree in the family Meliaceae, also known by the common names andiroba or crabwood.
Andiroba is native to the Amazon and is widely used by the indigenous populations of the northern region of Brazil. It grows in the Amazon region, Central America and the Caribbean. It is a tall tree with dense foliage and usually grows in the tropical rainforest along the edge of rivers. 
''', "https://en.wikipedia.org/wiki/Carapa_guianensis"),
    SpeciesDetail('Cariniana estrellensis', '''
Cariniana estrellensis is a species of rainforest tree in the family Lecythidaceae. It is native to South America. These trees can grow to extraordinary size. Perhaps the largest rainforest tree ever measured by college trained forester was a C. estrellensis measured by Edmundo Navarro de Andrade which was twenty-three feet thick (twenty-two meters girth) with no buttresses or basal swelling. 
''', "https://en.wikipedia.org/wiki/Cariniana_estrellensis"),
    SpeciesDetail('Cedrela fissilis', '''
Cedrela fissilis is a species of tree in the family Meliaceae. It is native to Central and South America, where it is distributed from Costa Rica to Argentina. Its common names include Argentine cedar, cedro batata, cedro blanco, "Acaju-catinga" (its Global Trees entry) and cedro colorado.
Once a common lowland forest tree, this species has been overexploited for timber and is now considered to be endangered. A few populations are stable, but many have been reduced, fragmented, and extirpated. The wood is often sold in batches with Cuban cedar (Cedrela odorata). 
''', "https://en.wikipedia.org/wiki/Cedrela_fissilis"),
    SpeciesDetail('Cedrelinga catenaeformis', '''
Cedrelinga is a genus of trees in the family Fabaceae. The only accepted species is Cedrelinga cateniformis, called tornillo or cedrorana, which is native to South America. It is occasionally harvested for its straight-grained timber.
''', "https://en.wikipedia.org/wiki/Cedrelinga"),
    SpeciesDetail('Cordia goeldiana', '''
It is often about 10 to 20 m in height, with a trunk diameter of about 40 to 60 cm. Heights of over 30 m and trunk diameters of up to 90 cm are also possible.
Cordia goeldiana is usually found in the lower Amazon in terra firme forests. It is reported in primary forests, although it also develops well under exposed conditions.
Freijo is reported to occur in Para and in the Tocantins and Xingu River basins of Brazil.
''', "http://www.tropicaltimber.info/specie/freijo-cordia-goeldiana/"),
    SpeciesDetail('Cordia sp', '''
Cordia is a genus of flowering plants in the borage family, Boraginaceae. It contains about 300 species of shrubs and trees, that are found worldwide, mostly in warmer regions. Many of the species are commonly called manjack, while bocote may refer to several Central American species in Spanish.
The generic name honours German botanist and pharmacist Valerius Cordus (1515–1544). Like most other Boraginaceae, the majority have trichomes (hairs) on the leaves. 
''', "https://en.wikipedia.org/wiki/Cordia"),
    SpeciesDetail('Couratari sp', '''
Couratari is a genus of trees in the family Lecythidaceae, first described as a genus in 1775. They are native to tropical South America and Central America.
They are large trees, often rising above the rainforest canopy. The leaves are evergreen, alternate, simple, elliptical, up to 15 cm long, with a serrate to serrulate margin. Vernation lines parallel to the midvein are often visible - a very unusual characteristic. The fruit is 6–15 cm long, and roughly conical. A central plug drops out at maturity, releasing the winged seeds to be dispersed by wind. The fruit of Cariniana may be distinguished from those of Couratari, as the former have longitudinal ridges, whereas the latter bears a single calyx-derived ring near the fruit apex. 
''', "https://en.wikipedia.org/wiki/Couratari"),
    SpeciesDetail('Dipteryx sp', '''
Dipteryx is a genus containing a number of species of large trees and possibly shrubs. It belongs to the "papilionoid" subfamily – Faboideae – of the family Fabaceae. This genus is native to South and Central America and the Caribbean. Formerly, the related genus Taralea was included in Dipteryx. 
''', "https://en.wikipedia.org/wiki/Dipteryx"),
    SpeciesDetail('Erisma uncinatum', '''
Erisma uncinatum is a species of tree in the family Vochysiaceae. They have a self-supporting growth form. They are native to Amapá, Mato Grosso, Maranhao, Amazônia, RondôNia, Pará, Acre (Brazil), and The Neotropics. They have simple, broad leaves. Individuals can grow to 27 m.
''', "https://eol.org/pages/5497350"),
    SpeciesDetail('Eucalyptus sp', '''
Eucalyptus (/ˌjuːkəˈlɪptəs/) is a genus of more than 700 species of flowering plants in the family Myrtaceae. Most species of Eucalyptus are trees, often mallees, and a few are shrubs. Along with several other genera in the tribe Eucalypteae, including Corymbia and Angophora, they are commonly known as eucalypts or "gum trees". Plants in the genus Eucalyptus have bark that is either smooth, fibrous, hard or stringy, the leaves have oil glands, and the sepals and petals are fused to form a "cap" or operculum over the stamens. The fruit is a woody capsule commonly referred to as a "gumnut". 
''', "https://en.wikipedia.org/wiki/Eucalyptus"),
    SpeciesDetail('Euxylophora paraensis', '''
Euxylophora paraensis is a species of tree in the family Rutaceae. They have a self-supporting growth form. They are listed as endangered by IUCN. They are native to Amazônia and South America. They have compound, broad leaves.
''', "https://eol.org/pages/5624064"),
    SpeciesDetail('Goupia glabra', '''
Goupia glabra (goupie or kabukalli; syn. G. paraensis, G. tomentosa) is a species of flowering plant in the family Goupiaceae (formerly treated in the family Celastraceae). It is native to tropical South America, in northern Brazil, Colombia, French Guiana, Guyana, Suriname, and Venezuela.
Other names include Saino, Sapino (Colombia), Kopi (Surinam), Kabukalli (Guyana), Goupi, bois-caca (French Guiana), Pasisi (Wayampi language), Pasis (Palikur language), Kopi (Businenge language), Cupiuba (Brazil), yãpi mamo hi (Yanomami language), Venezuela. 
''', "https://en.wikipedia.org/wiki/Goupia_glabra"),
    SpeciesDetail('Grevilea robusta', '''
Grevillea robusta, commonly known as the southern silky oak, silk oak or silky oak, silver oak or Australian silver oak, is a flowering plant in the family Proteaceae, and accordingly unrelated to true oaks, family Fagaceae. Grevillea robusta is a tree, and is the largest species in its genus. It is a native of eastern coastal Australia, growing in riverine, subtropical and dry rainforest environments. 
''', "https://en.wikipedia.org/wiki/Grevillea_robusta"),
    SpeciesDetail('Hura crepitans', '''
Hura crepitans, the sandbox tree, also known as possumwood, monkey no-climb, assacu (from Tupi asaku) and jabillo, is an evergreen tree in the family Euphorbiaceae, native to tropical regions of North and South America including the Amazon rainforest. It is also present in parts of Tanzania, where it is considered an invasive species. Because its fruits explode when ripe, it has also received the colloquial nickname the dynamite tree.
''', "https://en.wikipedia.org/wiki/Hura_crepitans"),
    SpeciesDetail('Hymenaea sp', '''
Hymenaea is a genus of plants in the legume family Fabaceae. Of the fourteen living species in the genus, all but one are native to the tropics of the Americas, with one additional species (Hymenaea verrucosa) on the east coast of Africa. Some authors place the African species in a separate monotypic genus, Trachylobium. In the Neotropics, Hymenaea is distributed through the Caribbean islands, and from southern Mexico to Brazil. Linnaeus named the genus in 1753 in Species Plantarum for Hymenaios, the Greek god of marriage ceremonies. The name is a reference to the paired leaflets. 
''', "https://en.wikipedia.org/wiki/Hymenaea"),
    SpeciesDetail('Hymenolobium petraeum', '''
Hymenolobium petraeum is a species of tree in the family legumes. They have a self-supporting growth form. They are native to Amazônia, Amapá, Pará, and Maranhao. They have compound, broad leaves.
''', "https://eol.org/pages/417255"),
    SpeciesDetail('Laurus nobilis', '''
Laurus nobilis /ˈlɔːrəs ˈnɒbɪlɪs/ is an aromatic evergreen tree or large shrub with green, glabrous (smooth) leaves. It is in the flowering plant family Lauraceae. It is native to the Mediterranean region and is used as bay leaf for seasoning in cooking. Its common names include bay tree (esp. United Kingdom),: 84  bay laurel, sweet bay, true laurel, Grecian laurel, or simply laurel. Laurus nobilis figures prominently in classical Greco-Roman culture.
Worldwide, many other kinds of plants in diverse families are also called "bay" or "laurel", generally due to similarity of foliage or aroma to Laurus nobilis. 
''', "https://en.wikipedia.org/wiki/Laurus_nobilis"),
    SpeciesDetail('Machaerium sp', '''
Machaerium is a genus of flowering plants in the family Fabaceae, and was recently assigned to the informal monophyletic Dalbergia clade of the Dalbergieae.
''', "https://en.wikipedia.org/wiki/Machaerium_(plant)"),
    SpeciesDetail('Manilkara huberi', '''
Manilkara huberi, also known as masaranduba, níspero, and sapotilla, is a fruit bearing plant of the genus Manilkara of the family Sapotaceae. 
''', "https://en.wikipedia.org/wiki/Manilkara_huberi"),
    SpeciesDetail('Melia azedarach', '''
Melia azedarach, commonly known as the chinaberry tree, pride of India, bead-tree, Cape lilac, syringa berrytree, Persian lilac, Indian lilac, or white cedar, is a species of deciduous tree in the mahogany family, Meliaceae, that is native to Indomalaya and Australasia.
''', "https://en.wikipedia.org/wiki/Melia_azedarach"),
    SpeciesDetail('Mezilaurus itauba', '''
Mezilaurus itauba is a species of tree in the family Lauraceae. It is found in Bolivia, Brazil, Ecuador, French Guiana, Peru, and Suriname.
''', "https://en.wikipedia.org/wiki/Mezilaurus_itauba"),
    SpeciesDetail('Micropholis venulosa', '''
Micropholis venulosa is a species of tree in the family Sapotaceae. They have a self-supporting growth form. They are native to Bahia, Mato Grosso, Pará, Maranhao, Espirito Santo, GoiáS, Cerrado, Amazônia, Distrito Federal, Mata Atlântica, Minas Gerais, Acre (Brazil), Mato Grosso Do Sul, RondôNia, and The Neotropics. They have simple, broad leaves. Individuals can grow to 21 m.
''', "https://eol.org/pages/1154259"),
    SpeciesDetail('Mimosa scabrella', '''
Mimosa scabrella is a tree in the family Fabaceae. It is very fast-growing and it can reach a height of 15 m (49 ft) tall in only 3 years. Its trunk is about 0.1–0.5 m (3.9–19.7 in) in diameter. It has yellow flowers.
''', "https://en.wikipedia.org/wiki/Mimosa_scabrella"),
    SpeciesDetail('Myroxylon balsamum', '''
Myroxylon balsamum, Santos mahogany, is a species of tree in the family Fabaceae. It is native to tropical forests from Southern Mexico through the Amazon regions of Peru and Brazil at elevations of 200–690 metres (660–2,260 ft). Plants are found growing in well drained soil in evergreen humid forest. 
''', "https://en.wikipedia.org/wiki/Myroxylon_balsamum"),
    SpeciesDetail('Ocotea porosa', '''
Ocotea porosa, commonly called imbuia or Brazilian walnut, is a species of plant in the Lauraceae family. Its wood is very hard, and it is a major commercial timber species in Brazil. 
''', "https://en.wikipedia.org/wiki/Ocotea_porosa"),
    SpeciesDetail('Peltogyne sp', '''
Peltogyne, commonly known as purpleheart, violet wood, amaranth and other local names (often referencing the colour of the wood) is a genus of 23 species of flowering plants in the family Fabaceae; native to tropical rainforests of Central and South America; from Guerrero, Mexico, through Central America, and as far as south-eastern Brazil.
They are medium-sized to large trees growing to 30–50 m (100–160 ft) tall, with trunk diameters of up to 1.5 m (5 ft). The leaves are alternate, divided into a symmetrical pair of large leaflets 5–10 cm (2–4 in) long and 2–4 cm (1–2 in) broad. The flowers are small, with five white petals, produced in panicles. The fruit is a pod containing a single seed. The timber is desirable, but difficult to work.
''', "https://en.wikipedia.org/wiki/Peltogyne"),
    SpeciesDetail('Pinus sp', '''
A pine is any conifer tree or shrub in the genus Pinus (/ˈpaɪnuːs/) of the family Pinaceae. Pinus is the sole genus in the subfamily Pinoideae.
World Flora Online, created by the Royal Botanic Gardens, Kew, and Missouri Botanical Garden, accepts 187 species names of pines as current, together with more synonyms. The American Conifer Society (ACS) and the Royal Horticultural Society accept 121 species.
Pines are commonly found in the Northern Hemisphere.
Pine may also refer to the lumber derived from pine trees; it is one of the more extensively used types of lumber.
The pine family is the largest conifer family, and there are currently 818 named cultivars (or trinomials) recognized by the ACS. It is also a well-known type of Christmas tree. 
''', "https://en.wikipedia.org/wiki/Pine"),
    SpeciesDetail('Podocarpus lambertii', '''
Podocarpus lambertii is a species of conifer in the family Podocarpaceae. It is found in Argentina and Brazil. 
''', "https://en.wikipedia.org/wiki/Podocarpus_lambertii"),
    SpeciesDetail('Pouteria pachycarpa', '''
A timber species from South and Central America. Commonly used in Heavy carpentry, Cabinetwork (high class furniture), Interior panelling, Turned goods, Flooring, Interior joinery, Sliced veneer, Tool handles (resilient woods)
''', "https://www.lesserknowntimberspecies.com/species/goiabao"),
    SpeciesDetail('Swietenia macrophylla', '''
Swietenia macrophylla, commonly known as mahogany, Honduran mahogany, Honduras mahogany, or big-leaf mahogany is a species of plant in the Meliaceae family. It is one of three species that yields genuine mahogany timber (Swietenia), the others being Swietenia mahagoni and Swietenia humilis. It is native to South America, Mexico and Central America, but naturalized in the Philippines, Singapore, Malaysia and Hawaii, and cultivated in plantations and wind-breaks elsewhere. 
''', "https://en.wikipedia.org/wiki/Swietenia_macrophylla"),
    SpeciesDetail('Tabebuia sp', '''
Tabebuia is a genus of flowering plants in the family Bignoniaceae. Tabebuia consists almost entirely of trees, but a few are often large shrubs. A few species produce timber, but the genus is mostly known for those that are cultivated as flowering trees. 
''', "https://en.wikipedia.org/wiki/Tabebuia"),
    SpeciesDetail('Virola surinamensis', '''
Virola surinamensis, known commonly as baboonwood, ucuuba, ucuhuba and chalviande, is a species of flowering plant in the family Myristicaceae. It is found in Brazil, Costa Rica, Ecuador, French Guiana, Guyana, Panama, Peru, Suriname, and Venezuela. It has also been naturalized in the Caribbean. Its natural habitats are subtropical or tropical moist lowland forests, subtropical or tropical swamps, and heavily degraded former forest. Although the species is listed as threatened due to habitat loss by the IUCN, it is a common tree species found throughout Central and South America.
''', "https://en.wikipedia.org/wiki/Virola_surinamensis"),
    SpeciesDetail('Vochysia sp', '''
Vochysia is a genus of plant in the family Vochysiaceae. It contains the following species, among many others:
Vochysia aurifera, Standl. & L.O. Williams
Vochysia haenkeana
Vochysia wilsonii Marc.-Berti, J.M. Vélez. & Aymard, 2023
''', "https://en.wikipedia.org/wiki/Vochysia"),
]