#!/usr/bin/env python

###################################### General Information ######################################

# Year to analyze
finalYear = 2016
mostRecentQuarter = '2017-09'
previousQuarter = '2016-09'

# WorldBank.org: https://data.worldbank.org/indicator/NY.GDP.MKTP.KD.ZG?end=2016&locations=CA&name_desc=true&start=2012
GDPGrowthRate = [2.475,2.565,0.942,1.468] # Canada

# Investing.com: http://www.investing.com/rates-bonds/
riskFreeRate = 2.266 / 100.0 # Canada (CAD 10Y Yield)

# WealthyEducation.com: https://wealthyeducation.com/resources/
marketRiskPremium = 4.77 / 100.0 # 

# Sectors + Industries
sectors = {
	'ConsumerDiscretionary': ['AutomobileComponents','Automobiles','Distributors','DiversifiedConsumerServices',
								'Hotels','Restaurants','Leisure','HouseholdDurables','InternetRetail','CatalogRetail',
								'LeisureProducts','Media','MultilineRetail','SpecialtyRetial','Textile',
								'Apparel','LuxuryGoods'],
	'ConsumerStaples': ['Beverages','FoodRetailing','StaplesRetailing','FoodProducts','HouseholdProducts',
						'PersonalProducts','Tobacco'],
	'Energy': ['EnergyEquipment','EnergyServices','Oil','Gas','ConsumableFuels'],
	'Financials': ['Banking','CapitalMarkets','ConsumerFinance','DiversifiedFinancialServices','Insurance',
					'REITs','RealEstateManagement','RealEstateDevelopment','RealEstate','Thrifts',
					'Mortgage'],
	'HealthCare': ['Biotechnology','HealthCareEquipment','HealthCareSupplies','HealthCareProviders',
					'HealthCareServices','HealthCareTechnology','LifeSciencesTool','LifeScienceServices',
					'Pharmaceuticals'],
	'Industrial': ['Aerospace','Defense','AirFreight','Logistics','Airlines','BuildingProducts',
					'CommercialServices','CommercialSupplies','Construction','Engineering',
					'ElectricalEquipment','IndustrialConglomerates','Machinery','Marine',
					'ProfessionalServices','Road','Rail','TradingCompanies','TradingDistributors',
					'TransportationInfrastructure'],
	'Information': ['CommunicationsEquipment','ElectronicEquipment','ElectronicInstruments','ElectronicComponents',
					'ITServices','InternetSoftware','InternetServices','Semiconductors','SemiconductorEquipment',
					'Software','HardwareTechnology','StorageTechnology','PeripheralTechnology'],
	'Materials': ['Chemicals','ConstructionMaterials','Containers','Packaging','Metals','Mining','PaperProducts','ForestProducts'],
	'Telecommunication': ['DiversifiedTelecommunication','WirelessTelecommunication','Telecommunication'],
	'Utilities': ['ElectricUtilities','GasUtilities','IndependentPowerProducers','RenewableElectricityProducers',
				  'MultiUtilitiesIndustry','WaterUtilities']
}

cyclical = ['BasicMaterial','ConsumerCyclical','FinancialServices','RealEstate','Automotive']
defensive = ['ConsumerGoods','HealthCare','Utilities']
sensitive = ['CommunicationServices','Energy','Industrial','Technology']

# Market Type
market = 'boom' # or 'fair'

# Tolerance for % difference between net income and revenue and profit
tol = 0.25