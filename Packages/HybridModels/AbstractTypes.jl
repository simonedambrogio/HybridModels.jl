abstract type AbstractHybridModel end
abstract type AbstractComponent <: AbstractHybridModel end
abstract type AbstractKnowledgeDrivenComponent <: AbstractComponent end
abstract type AbstractDataDrivenComponent <: AbstractComponent end